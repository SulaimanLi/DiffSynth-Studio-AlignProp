import torch, os, argparse, accelerate, copy
from diffsynth.core import UnifiedDataset
from diffsynth.pipelines.z_image import ZImagePipeline, ModelConfig
from diffsynth.diffusion import *
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class ZImageTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
        preference_loss_type="dpo",
        preference_beta=100.0,
        preference_num_timesteps=4,
        preference_loss_weight=1.0,
        preference_sft_weight=0.1,
        enable_npu_patch=True,
    ):
        super().__init__()
        if task.startswith("text_pref_dpo:"):
            raise NotImplementedError("Split cached training is not implemented for text_pref_dpo yet.")
        if task == "text_pref_dpo":
            if trainable_models not in (None, ""):
                raise ValueError("text_pref_dpo only supports LoRA training on `dit`. Leave `trainable_models` empty.")
            if lora_base_model != "dit":
                raise ValueError("text_pref_dpo requires `--lora_base_model dit`.")
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, device=device)
        tokenizer_config = ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="tokenizer/") if tokenizer_path is None else ModelConfig(tokenizer_path)
        self.pipe = ZImagePipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs, tokenizer_config=tokenizer_config, enable_npu_patch=enable_npu_patch)
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)

        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )
        
        # Other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.preference_loss_type = preference_loss_type
        self.reference_models = torch.nn.ModuleDict()
        self.reference_model_names = set()
        self.lora_base_model = lora_base_model
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
        }
        if task == "text_pref_dpo":
            self.loss_fn = FlowMatchPreferenceLoss(
                beta=preference_beta,
                num_timestep_samples=preference_num_timesteps,
                preference_loss_weight=preference_loss_weight,
                sft_loss_weight=preference_sft_weight,
                loss_type=preference_loss_type,
            )
            if preference_loss_type == "dpo":
                self.build_reference_models()
        if task == "trajectory_imitation":
            # This is an experimental feature.
            # We may remove it in the future.
            self.loss_fn = TrajectoryImitationLoss()
            self.task_to_loss["trajectory_imitation"] = self.loss_fn
            self.pipe_teacher = copy.deepcopy(self.pipe)
            self.pipe_teacher.requires_grad_(False)

    def build_reference_models(self):
        # text_pref_dpo only trains LoRA on the DiT branch.
        # Frozen modules can be shared with the current pipeline because their weights never change.
        if self.lora_base_model is None:
            return
        model = getattr(self.pipe, self.lora_base_model)
        if model is None:
            return
        reference_model = copy.deepcopy(model)
        reference_model.requires_grad_(False)
        reference_model.eval()
        self.reference_models[self.lora_base_model] = reference_model
        self.reference_model_names.add(self.lora_base_model)

    def get_reference_iteration_models(self):
        models = {}
        for name in self.pipe.in_iteration_models:
            if name in self.reference_model_names:
                models[name] = self.reference_models[name]
            else:
                models[name] = getattr(self.pipe, name)
        return models

    def build_shared_inputs(self, image):
        return {
            "input_image": image,
            "height": image.size[1],
            "width": image.size[0],
            "cfg_scale": 1,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
        }

    def build_branch_inputs(self, data, image_key="image"):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {"negative_prompt": ""}
        inputs_shared = self.build_shared_inputs(data[image_key])
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega
        
    def get_pipeline_inputs(self, data):
        inputs_shared, inputs_posi, inputs_nega = self.build_branch_inputs(data, image_key="image")
        if self.task == "trajectory_imitation":
            inputs_shared["cfg_scale"] = 2
            inputs_shared["teacher"] = self.pipe_teacher
        return inputs_shared, inputs_posi, inputs_nega

    def get_preference_inputs(self, data):
        required_keys = ("chosen_image", "rejected_image")
        missing_keys = [key for key in required_keys if key not in data]
        if len(missing_keys) > 0:
            raise KeyError(
                "Preference training expects metadata fields "
                "`chosen_image` and `rejected_image`. "
                f"Missing: {missing_keys}"
            )
        return {
            "chosen": self.build_branch_inputs(data, image_key="chosen_image"),
            "rejected": self.build_branch_inputs(data, image_key="rejected_image"),
        }

    def preprocess_pipeline_inputs(self, inputs):
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        return inputs

    def preprocess_preference_inputs(self, preference_inputs):
        chosen_inputs_shared, chosen_inputs_posi, _ = self.preprocess_pipeline_inputs(preference_inputs["chosen"])
        rejected_inputs_shared, rejected_inputs_posi, _ = self.preprocess_pipeline_inputs(preference_inputs["rejected"])
        return {
            "chosen": {**chosen_inputs_shared, **chosen_inputs_posi},
            "rejected": {**rejected_inputs_shared, **rejected_inputs_posi},
        }
    
    def forward(self, data, inputs=None):
        if self.task == "text_pref_dpo":
            if inputs is None:
                inputs = self.get_preference_inputs(data)
            inputs = self.preprocess_preference_inputs(inputs)
            reference_models = None if self.preference_loss_type != "dpo" else self.get_reference_iteration_models()
            return self.loss_fn(self.pipe, inputs, reference_models=reference_models)
        if inputs is None: inputs = self.get_pipeline_inputs(data)
        inputs = self.preprocess_pipeline_inputs(inputs)
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss


def z_image_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_image_size_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--enable_npu_patch", default=False, action="store_true", help="Whether to use npu fused operator patch to improve performance in NPU.")
    return parser


if __name__ == "__main__":
    parser = z_image_parser()
    args = parser.parse_args()
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_image_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
        )
    )
    model = ZImageTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        preference_loss_type=args.preference_loss_type,
        preference_beta=args.preference_beta,
        preference_num_timesteps=args.preference_num_timesteps,
        preference_loss_weight=args.preference_loss_weight,
        preference_sft_weight=args.preference_sft_weight,
        device=accelerator.device,
        enable_npu_patch=args.enable_npu_patch
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
        "direct_distill": launch_training_task,
        "direct_distill:train": launch_training_task,
        "text_pref_dpo": launch_training_task,
        "trajectory_imitation": launch_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
