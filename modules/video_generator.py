import torch
import numpy as np
from PIL import Image
import cv2
from typing import List, Optional, Tuple
import os
from modules.config import path_models
from modules.model_loader import load_model
from modules.util import get_model_path

class VideoGenerator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.fps = 24
        self.max_frames = 720  # 30 segundos a 24fps
        
    def load_models(self):
        """Carrega os modelos necessários para geração de vídeo"""
        try:
            # Carrega o modelo base
            self.model = load_model(
                model_path=get_model_path("juggernautXL_v8Rundiffusion.safetensors"),
                device=self.device
            )
            
            # Carrega o modelo de animação
            self.animate_model = load_model(
                model_path=get_model_path("animatediff_v1.5.3.safetensors"),
                device=self.device
            )
            
            return True
        except Exception as e:
            print(f"Erro ao carregar modelos: {str(e)}")
            return False
            
    def generate_video(
        self,
        prompt: str,
        negative_prompt: str,
        duration: int = 30,
        motion_type: str = "pan",
        motion_params: dict = None,
        output_path: str = "outputs/video.mp4"
    ) -> bool:
        """
        Gera um vídeo baseado no prompt fornecido
        
        Args:
            prompt: Texto descritivo do vídeo
            negative_prompt: Texto para elementos a evitar
            duration: Duração em segundos (máx 30)
            motion_type: Tipo de movimento ("pan", "zoom", "interpolation")
            motion_params: Parâmetros específicos do movimento
            output_path: Caminho para salvar o vídeo
            
        Returns:
            bool: True se o vídeo foi gerado com sucesso
        """
        try:
            # Limita a duração
            duration = min(duration, 30)
            num_frames = min(duration * self.fps, self.max_frames)
            
            # Gera frames base
            frames = self._generate_base_frames(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_frames=num_frames
            )
            
            # Aplica movimento
            if motion_type == "pan":
                frames = self._apply_pan(frames, motion_params)
            elif motion_type == "zoom":
                frames = self._apply_zoom(frames, motion_params)
            elif motion_type == "interpolation":
                frames = self._apply_interpolation(frames, motion_params)
                
            # Salva o vídeo
            self._save_video(frames, output_path)
            
            return True
            
        except Exception as e:
            print(f"Erro ao gerar vídeo: {str(e)}")
            return False
            
    def _generate_base_frames(
        self,
        prompt: str,
        negative_prompt: str,
        num_frames: int
    ) -> List[Image.Image]:
        """Gera frames base usando o modelo principal"""
        frames = []
        
        # Gera frames iniciais
        for i in range(num_frames):
            # Ajusta o prompt para cada frame
            frame_prompt = f"{prompt}, frame {i+1} of {num_frames}"
            
            # Gera a imagem
            image = self.model.generate(
                prompt=frame_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=30,
                guidance_scale=7.0
            )
            
            frames.append(image)
            
        return frames
        
    def _apply_pan(
        self,
        frames: List[Image.Image],
        params: Optional[dict] = None
    ) -> List[Image.Image]:
        """Aplica movimento de pan aos frames"""
        if params is None:
            params = {"direction": "right", "speed": 0.1}
            
        # Converte frames para numpy
        frames_np = [np.array(frame) for frame in frames]
        
        # Aplica pan
        for i in range(1, len(frames_np)):
            if params["direction"] == "right":
                shift = int(i * params["speed"] * frames_np[i].shape[1])
                frames_np[i] = np.roll(frames_np[i], shift, axis=1)
            elif params["direction"] == "left":
                shift = int(-i * params["speed"] * frames_np[i].shape[1])
                frames_np[i] = np.roll(frames_np[i], shift, axis=1)
                
        return [Image.fromarray(frame) for frame in frames_np]
        
    def _apply_zoom(
        self,
        frames: List[Image.Image],
        params: Optional[dict] = None
    ) -> List[Image.Image]:
        """Aplica zoom aos frames"""
        if params is None:
            params = {"zoom_factor": 1.5, "zoom_speed": 0.01}
            
        # Converte frames para numpy
        frames_np = [np.array(frame) for frame in frames]
        
        # Aplica zoom
        for i in range(1, len(frames_np)):
            zoom = 1 + (i * params["zoom_speed"])
            zoom = min(zoom, params["zoom_factor"])
            
            h, w = frames_np[i].shape[:2]
            new_h, new_w = int(h * zoom), int(w * zoom)
            
            # Redimensiona
            zoomed = cv2.resize(frames_np[i], (new_w, new_h))
            
            # Recorta para o tamanho original
            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            frames_np[i] = zoomed[start_h:start_h+h, start_w:start_w+w]
            
        return [Image.fromarray(frame) for frame in frames_np]
        
    def _apply_interpolation(
        self,
        frames: List[Image.Image],
        params: Optional[dict] = None
    ) -> List[Image.Image]:
        """Aplica interpolação entre frames"""
        if params is None:
            params = {"interpolation_factor": 2}
            
        # Converte frames para numpy
        frames_np = [np.array(frame) for frame in frames]
        
        # Interpola frames
        interpolated = []
        for i in range(len(frames_np) - 1):
            interpolated.append(frames_np[i])
            
            for j in range(1, params["interpolation_factor"]):
                alpha = j / params["interpolation_factor"]
                interp_frame = cv2.addWeighted(
                    frames_np[i],
                    1 - alpha,
                    frames_np[i + 1],
                    alpha,
                    0
                )
                interpolated.append(interp_frame)
                
        interpolated.append(frames_np[-1])
        
        return [Image.fromarray(frame) for frame in interpolated]
        
    def _save_video(
        self,
        frames: List[Image.Image],
        output_path: str
    ):
        """Salva os frames como vídeo MP4"""
        # Cria diretório se não existir
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Obtém dimensões do primeiro frame
        height, width = np.array(frames[0]).shape[:2]
        
        # Configura o writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            output_path,
            fourcc,
            self.fps,
            (width, height)
        )
        
        # Salva frames
        for frame in frames:
            out.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
            
        out.release() 