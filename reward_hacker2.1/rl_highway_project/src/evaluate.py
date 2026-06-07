import gymnasium as gym
import os
import imageio
import numpy as np
from stable_baselines3 import PPO

# 1. KESİN ÇÖZÜM: Kendi eğitim ayarlarımızı projeden çekiyoruz!
from config import ProjectConfig

if "SDL_VIDEODRIVER" in os.environ:
    del os.environ["SDL_VIDEODRIVER"]

def record_eval_video(model_path, output_dir, video_name):
    print(f"\nDeğerlendiriliyor: {model_path} -> Kayıt yeri: {output_dir}")
    
    # Eğitimdeki birebir aynı ortam ayarlarını yükle
    proj_config = ProjectConfig()
    env_dict = proj_config.env_config.to_dict()
    
    # Sadece video çekimi için gereken görsel ayarları üzerine yaz (override)
    env_dict.update({
        "manual_control": False,
        "screen_width": 800,
        "screen_height": 200,
        "duration": 200
    })
    
    env = gym.make("highway-v0", render_mode="rgb_array")
    env.unwrapped.configure(env_dict)
    
    if not os.path.exists(f"{model_path}.zip"):
        print(f"HATA: '{model_path}.zip' dosyası bulunamadı!")
        return
        
    model = PPO.load(model_path)
    obs, info = env.reset()
    
    frames = []
    done = truncated = False
    
    # Otoyol modundaki standart komutların isimleri (Terminalde görmek için)
    action_names = {0: "SOL_ŞERİDE_GEÇ", 1: "SABİT_GİT (Rölanti)", 2: "SAĞ_ŞERİDE_GEÇ", 3: "GAZA_BAS", 4: "FRENE_BAS"}
    
    print("Kamera kayıtta... AI kararları ekrana yazdırılıyor:")
    
    while not (done or truncated) and len(frames) < 200:
        # AI Kararı alıyor
        action, _ = model.predict(obs, deterministic=True)
        
        # 2. KRİTİK NOKTA: Yapay zekanın karmaşık formatını saf sayıya çevir (Tekerleğe ilet)
        pure_action = int(action.item())
        
        # Terminalde yapay zekanın beynini oku (ilk 10 adım için)
        if len(frames) < 10:
            print(f"Adım {len(frames)} -> AI Beyni: {action_names.get(pure_action, pure_action)}")
            
        obs, reward, done, truncated, info = env.step(pure_action)
        
        frame = env.render()
        if frame is not None and isinstance(frame, np.ndarray) and frame.max() > 0:
            frames.append(frame)
            
    env.close()
        
    os.makedirs(output_dir, exist_ok=True)
    video_path = os.path.join(output_dir, f"{video_name}.mp4")
    
    imageio.mimsave(video_path, frames, fps=15)
    print(f"✅ Video başarıyla kaydedildi ({len(frames)} kare): {video_path}")

if __name__ == "__main__":
    models_to_evaluate = [
        {"path": "checkpoints/model_start", "out": "assets/start", "video_name": "start_episode"},
        {"path": "checkpoints/model_halfway", "out": "assets/halfway", "video_name": "halfway_episode"},
        {"path": "checkpoints/model_final", "out": "assets/final", "video_name": "final_episode"}
    ]
    
    for m in models_to_evaluate:
        record_eval_video(m["path"], m["out"], m["video_name"])