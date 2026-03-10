from .base_pipeline import BasePipeline
import torch


def _sample_flow_match_timestep_ids(pipe: BasePipeline, inputs, num_timestep_samples=1):
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))
    timestep_ids = torch.randint(min_timestep_boundary, max_timestep_boundary, (num_timestep_samples,))
    return timestep_ids


def _prepare_flow_match_noise(input_latents, num_timestep_samples=1):
    return [torch.randn_like(input_latents) for _ in range(num_timestep_samples)]


def _flow_match_loss_from_samples(pipe: BasePipeline, models, inputs, timestep_ids, noises):
    losses = []
    for timestep_id, noise in zip(timestep_ids, noises):
        timestep = pipe.scheduler.timesteps[timestep_id].unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)

        inputs_ = dict(inputs)
        inputs_["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
        training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)

        if "first_frame_latents" in inputs_:
            inputs_["latents"][:, :, 0:1] = inputs_["first_frame_latents"]

        noise_pred = pipe.model_fn(**models, **inputs_, timestep=timestep)

        if "first_frame_latents" in inputs_:
            noise_pred = noise_pred[:, :, 1:]
            training_target = training_target[:, :, 1:]

        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        loss = loss * pipe.scheduler.training_weight(timestep)
        losses.append(loss.reshape(()))
    return torch.stack(losses).mean()


def FlowMatchLoss(pipe: BasePipeline, models=None, num_timestep_samples=1, timestep_ids=None, noises=None, **inputs):
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models} if models is None else models
    timestep_ids = _sample_flow_match_timestep_ids(pipe, inputs, num_timestep_samples) if timestep_ids is None else timestep_ids
    noises = _prepare_flow_match_noise(inputs["input_latents"], len(timestep_ids)) if noises is None else noises
    return _flow_match_loss_from_samples(pipe, models, inputs, timestep_ids, noises)


def FlowMatchSFTLoss(pipe: BasePipeline, **inputs):
    return FlowMatchLoss(pipe, **inputs)


def FlowMatchSFTAudioVideoLoss(pipe: BasePipeline, **inputs):
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
    
    # video
    noise = torch.randn_like(inputs["input_latents"])
    inputs["video_latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)
    
    # audio
    if inputs.get("audio_input_latents") is not None:
        audio_noise = torch.randn_like(inputs["audio_input_latents"])
        inputs["audio_latents"] = pipe.scheduler.add_noise(inputs["audio_input_latents"], audio_noise, timestep)
        training_target_audio = pipe.scheduler.training_target(inputs["audio_input_latents"], audio_noise, timestep)

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred, noise_pred_audio = pipe.model_fn(**models, **inputs, timestep=timestep)

    loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
    loss = loss * pipe.scheduler.training_weight(timestep)
    if inputs.get("audio_input_latents") is not None:
        loss_audio = torch.nn.functional.mse_loss(noise_pred_audio.float(), training_target_audio.float())
        loss_audio = loss_audio * pipe.scheduler.training_weight(timestep)
        loss = loss + loss_audio
    return loss


def DirectDistillLoss(pipe: BasePipeline, **inputs):
    pipe.scheduler.set_timesteps(inputs["num_inference_steps"])
    pipe.scheduler.training = True
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
        timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
        noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep, progress_id=progress_id)
        inputs["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs)
    loss = torch.nn.functional.mse_loss(inputs["latents"].float(), inputs["input_latents"].float())
    return loss


class FlowMatchPreferenceLoss(torch.nn.Module):
    def __init__(
        self,
        beta=100.0,
        num_timestep_samples=4,
        preference_loss_weight=1.0,
        sft_loss_weight=0.1,
        loss_type="dpo",
    ):
        super().__init__()
        self.beta = beta
        self.num_timestep_samples = num_timestep_samples
        self.preference_loss_weight = preference_loss_weight
        self.sft_loss_weight = sft_loss_weight
        self.loss_type = loss_type

    def _sample_branch_noises(self, chosen_inputs, rejected_inputs):
        chosen_noises = _prepare_flow_match_noise(chosen_inputs["input_latents"], self.num_timestep_samples)
        if chosen_inputs["input_latents"].shape == rejected_inputs["input_latents"].shape:
            rejected_noises = [noise.clone() for noise in chosen_noises]
        else:
            rejected_noises = _prepare_flow_match_noise(rejected_inputs["input_latents"], self.num_timestep_samples)
        return chosen_noises, rejected_noises

    def _preference_logits(self, current_losses, reference_losses):
        chosen_current_loss, rejected_current_loss = current_losses
        chosen_reference_loss, rejected_reference_loss = reference_losses
        if self.loss_type == "dpo":
            return (rejected_current_loss - chosen_current_loss) - (rejected_reference_loss - chosen_reference_loss)
        elif self.loss_type == "ranking":
            return rejected_current_loss - chosen_current_loss
        else:
            raise ValueError(f"Unsupported preference loss type: {self.loss_type}")

    def forward(self, pipe: BasePipeline, preference_inputs, reference_models=None):
        chosen_inputs = preference_inputs["chosen"]
        rejected_inputs = preference_inputs["rejected"]

        timestep_ids = _sample_flow_match_timestep_ids(pipe, chosen_inputs, self.num_timestep_samples)
        chosen_noises, rejected_noises = self._sample_branch_noises(chosen_inputs, rejected_inputs)

        current_models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        chosen_current_loss = FlowMatchLoss(
            pipe,
            models=current_models,
            timestep_ids=timestep_ids,
            noises=chosen_noises,
            **chosen_inputs,
        )
        rejected_current_loss = FlowMatchLoss(
            pipe,
            models=current_models,
            timestep_ids=timestep_ids,
            noises=rejected_noises,
            **rejected_inputs,
        )

        if self.loss_type == "dpo":
            if reference_models is None:
                raise ValueError("reference_models is required when preference_loss_type is dpo.")
            with torch.no_grad():
                chosen_reference_loss = FlowMatchLoss(
                    pipe,
                    models=reference_models,
                    timestep_ids=timestep_ids,
                    noises=chosen_noises,
                    **chosen_inputs,
                )
                rejected_reference_loss = FlowMatchLoss(
                    pipe,
                    models=reference_models,
                    timestep_ids=timestep_ids,
                    noises=rejected_noises,
                    **rejected_inputs,
                )
        else:
            chosen_reference_loss = torch.zeros_like(chosen_current_loss)
            rejected_reference_loss = torch.zeros_like(rejected_current_loss)

        preference_logits = self._preference_logits(
            (chosen_current_loss, rejected_current_loss),
            (chosen_reference_loss, rejected_reference_loss),
        )
        preference_loss = -torch.nn.functional.logsigmoid(self.beta * preference_logits)
        loss = self.preference_loss_weight * preference_loss
        if self.sft_loss_weight > 0:
            loss = loss + self.sft_loss_weight * chosen_current_loss
        return loss


class TrajectoryImitationLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.initialized = False
    
    def initialize(self, device):
        import lpips # TODO: remove it
        self.loss_fn = lpips.LPIPS(net='alex').to(device)
        self.initialized = True

    def fetch_trajectory(self, pipe: BasePipeline, timesteps_student, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        trajectory = [inputs_shared["latents"].clone()]

        pipe.scheduler.set_timesteps(num_inference_steps, target_timesteps=timesteps_student)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred.detach(), **inputs_shared)

            trajectory.append(inputs_shared["latents"].clone())
        return pipe.scheduler.timesteps, trajectory
    
    def align_trajectory(self, pipe: BasePipeline, timesteps_teacher, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        loss = 0
        pipe.scheduler.set_timesteps(num_inference_steps, training=True)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)

            progress_id_teacher = torch.argmin((timesteps_teacher - timestep).abs())
            inputs_shared["latents"] = trajectory_teacher[progress_id_teacher]

            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )

            sigma = pipe.scheduler.sigmas[progress_id]
            sigma_ = 0 if progress_id + 1 >= len(pipe.scheduler.timesteps) else pipe.scheduler.sigmas[progress_id + 1]
            if progress_id + 1 >= len(pipe.scheduler.timesteps):
                latents_ = trajectory_teacher[-1]
            else:
                progress_id_teacher = torch.argmin((timesteps_teacher - pipe.scheduler.timesteps[progress_id + 1]).abs())
                latents_ = trajectory_teacher[progress_id_teacher]
            
            denom = sigma_ - sigma
            denom = torch.sign(denom) * torch.clamp(denom.abs(), min=1e-6)
            target = (latents_ - inputs_shared["latents"]) / denom
            loss = loss + torch.nn.functional.mse_loss(noise_pred.float(), target.float()) * pipe.scheduler.training_weight(timestep)
        return loss
    
    def compute_regularization(self, pipe: BasePipeline, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        inputs_shared["latents"] = trajectory_teacher[0]
        pipe.scheduler.set_timesteps(num_inference_steps)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred.detach(), **inputs_shared)

        image_pred = pipe.vae_decoder(inputs_shared["latents"])
        image_real = pipe.vae_decoder(trajectory_teacher[-1])
        loss = self.loss_fn(image_pred.float(), image_real.float())
        return loss

    def forward(self, pipe: BasePipeline, inputs_shared, inputs_posi, inputs_nega):
        if not self.initialized:
            self.initialize(pipe.device)
        with torch.no_grad():
            pipe.scheduler.set_timesteps(8)
            timesteps_teacher, trajectory_teacher = self.fetch_trajectory(inputs_shared["teacher"], pipe.scheduler.timesteps, inputs_shared, inputs_posi, inputs_nega, 50, 2)
            timesteps_teacher = timesteps_teacher.to(dtype=pipe.torch_dtype, device=pipe.device)
        loss_1 = self.align_trajectory(pipe, timesteps_teacher, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, 8, 1)
        loss_2 = self.compute_regularization(pipe, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, 8, 1)
        loss = loss_1 + loss_2
        return loss
