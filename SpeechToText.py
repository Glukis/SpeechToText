"""
Voice Recorder with Mouse Control - Simplified Version
Holds middle mouse button (wheel) for 0.2+ second to record audio, releases to transcribe with Whisper
Works globally - no need to focus the program window
Audio is processed directly in memory without saving to disk
Features: Sound notifications, auto-paste to active field (NO VISUAL WINDOW - more stable)
"""

import pyaudio
import threading
import time
import whisper
import numpy as np
from pynput import mouse
from pynput.mouse import Button, Listener
import winsound
import pyperclip
import pyautogui
import torch
import keyboard


class VoiceRecorder:
    def __init__(self, model_size="base", use_cuda=True, language="ru", mic_gain=1.0, playback_before_transcribe=False, playback_volume=3.0):
        """
        Initialize the Voice Recorder
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            use_cuda: Use CUDA for GPU acceleration if available
            language: Language code for transcription (ru, en, de, fr, es, etc.)
            mic_gain: Microphone gain/sensitivity (0.1 to 5.0, default 1.0)
                     1.0 = normal, <1.0 = quieter, >1.0 = louder
            playback_before_transcribe: If True, plays back recording before transcription
            playback_volume: Volume multiplier for playback (1.0 to 10.0, default 3.0)
                            Higher values = louder playback
        """
        self.is_recording = False
        self.audio_frames = []
        self.press_time = None
        self.recording_thread = None
        self.recording_started = False  # Flag to prevent double-start
        self.language = language  # Store language setting
        self.mic_gain = max(0.1, min(5.0, mic_gain))  # Clamp between 0.1 and 5.0
        self.playback_before_transcribe = playback_before_transcribe  # Playback flag
        self.playback_volume = max(1.0, min(10.0, playback_volume))  # Clamp between 1.0 and 10.0
        
        # Audio settings
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.audio = pyaudio.PyAudio()
        self.stream = None
        # Check CUDA availability
        self.device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        self.use_fp16 = (self.device == "cuda")

        # Load Whisper model
        print(f"Loading Whisper model ({model_size})...")
        print(f"[*] Device: {self.device.upper()}")
        print(f"[*] Language: {self.get_language_name(language)}")
        print(f"[*] Mic Gain: {self.mic_gain}x", end="")
        if self.mic_gain < 1.0:
            print(" (quieter)")
        elif self.mic_gain > 1.0:
            print(" (louder)")
        else:
            print(" (normal)")
        
        if self.playback_before_transcribe:
            print(f"[*] Playback Preview: ENABLED (volume: {self.playback_volume}x)")
        else:
            print(f"[*] Playback Preview: DISABLED")
        
        if self.device == "cuda":
            try:
                gpu_name = torch.cuda.get_device_name(0)
                print(f"[*] GPU: {gpu_name}")
                print(f"[*] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
                print(f"[*] CUDA Version: {torch.version.cuda}")
            except:
                pass
        else:
            print("[!] CUDA not available, using CPU")
            if not torch.cuda.is_available():
                print("    To enable GPU: install CUDA and PyTorch with CUDA support")
        
        self.model = whisper.load_model(model_size, device=self.device)
        print("[OK] Model loaded successfully!")
        print()
    
    def get_language_name(self, code):
        """Get full language name from code"""
        languages = {
            "ru": "Russian (Русский)",
            "en": "English",
            "de": "German (Deutsch)",
            "fr": "French (Français)",
            "es": "Spanish (Español)",
            "it": "Italian (Italiano)",
            "pt": "Portuguese (Português)",
            "pl": "Polish (Polski)",
            "uk": "Ukrainian (Українська)",
            "ja": "Japanese (日本語)",
            "zh": "Chinese (中文)",
            "ko": "Korean (한국어)",
            "ar": "Arabic (العربية)",
            "hi": "Hindi (हिन्दी)",
        }
        return languages.get(code, f"{code.upper()}")
    
    def play_sound(self, frequency=800, duration=100):
        """Play a beep sound"""
        try:
            winsound.Beep(frequency, duration)
        except:
            print('\a')  # Fallback to system beep
    
    def playback_recording(self):
        """Play back the recorded audio for preview"""
        if not self.audio_frames:
            print("⚠️  No audio frames to play")
            return
        
        duration = len(self.audio_frames) * self.CHUNK / self.RATE
        print(f"🔊 Playing back recording ({len(self.audio_frames)} frames, ~{duration:.1f}s, volume: {self.playback_volume}x)...")
        
        try:
            # Convert all frames to numpy array first
            import numpy as np
            all_data = b''.join(self.audio_frames)
            audio_array = np.frombuffer(all_data, dtype=np.int16)
            
            # Show audio levels for debugging
            print(f"   📊 Audio levels: min={audio_array.min()}, max={audio_array.max()}, rms={np.sqrt(np.mean(audio_array.astype(float)**2)):.1f}")
            
            # Apply volume boost
            boosted = audio_array.astype(np.float32) * self.playback_volume
            
            # Clip to prevent distortion
            boosted = np.clip(boosted, -32768, 32767).astype(np.int16)
            
            # Open stream for playback
            playback_stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                output=True,
                frames_per_buffer=self.CHUNK
            )
            
            # Play in chunks
            for i in range(0, len(boosted), self.CHUNK):
                chunk = boosted[i:i+self.CHUNK]
                playback_stream.write(chunk.tobytes())
            
            # Ensure all data is played
            time.sleep(0.1)
            
            # Close playback stream
            playback_stream.stop_stream()
            playback_stream.close()
            
            print("[OK] Playback finished")
            time.sleep(0.2)  # Short pause before transcription
            
        except Exception as e:
            print(f"[!] Playback error: {e}")
            import traceback
            traceback.print_exc()
            # Continue with transcription even if playback fails
    
    def start_recording(self):
        """Start recording audio from microphone"""
        self.audio_frames = []
        self.stream = self.audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK
        )
        self.is_recording = True
        
        # Play start sound
        threading.Thread(target=self.play_sound, args=(1000, 150), daemon=True).start()
        
        print("[REC] Recording started...")
        
        while self.is_recording:
            try:
                data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                self.audio_frames.append(data)
            except Exception as e:
                print(f"Error during recording: {e}")
                break
    
    def stop_recording(self):
        """Stop recording and transcribe audio"""
        if self.is_recording:
            self.is_recording = False
            if self.recording_thread:
                self.recording_thread.join()
            
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            
            print("[STOP] Recording stopped")
            
            # Playback recording if enabled
            if self.audio_frames:
                if self.playback_before_transcribe:
                    self.playback_recording()
                
                # Transcribe audio
                self.transcribe_audio()
    
    def transcribe_audio(self):
        """Transcribe audio using Whisper directly from memory"""
        print("[AI] Transcribing audio...")
        try:
            # Convert audio frames to numpy array
            audio_data = b''.join(self.audio_frames)
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Apply microphone gain
            if self.mic_gain != 1.0:
                audio_np = audio_np * self.mic_gain
                # Clip to prevent distortion
                audio_np = np.clip(audio_np, -1.0, 1.0)
            
            # Transcribe directly from numpy array with GPU acceleration
            result = self.model.transcribe(
                audio_np, 
                language=self.language,  # Use configured language
                fp16=self.use_fp16  # Use FP16 on CUDA for speed
            )
            text = result["text"].strip()
            
            if text:
                print("\n" + "="*50)
                print("[TEXT] Transcription:")
                print(text)
                print("="*50 + "\n")
                
                # Copy to clipboard WITHOUT trailing space
                pyperclip.copy(text)
                print("[COPY] Text copied to clipboard!")
                
                # Auto-type text directly to active field (more reliable than paste)
                time.sleep(0.3)  # Delay to ensure clipboard is ready and focus is set
                paste_success = False
                
                # Try method 1: Direct typing with keyboard library (most reliable)
                try:
                    print("[PASTE] Attempting direct typing (method 1: keyboard)...")
                    keyboard.write(text + ' ')  # Type text directly with trailing space
                    paste_success = True
                    print("[PASTE] ✓ Text automatically typed to active field (with trailing space)!")
                except Exception as e:
                    print(f"[!] Method 1 failed: {e}")
                
                # Try method 2: pyautogui typing as fallback
                if not paste_success:
                    try:
                        print("[PASTE] Attempting direct typing (method 2: pyautogui)...")
                        pyautogui.write(text + ' ', interval=0.01)  # Type with small delay between chars
                        paste_success = True
                        print("[PASTE] ✓ Text automatically typed to active field (with trailing space)!")
                    except Exception as e:
                        print(f"[!] Method 2 failed: {e}")
                
                # Try method 3: Clipboard paste with Ctrl+V as last resort
                if not paste_success:
                    try:
                        print("[PASTE] Attempting clipboard paste (method 3: ctrl+v)...")
                        keyboard.press_and_release('ctrl+v')
                        time.sleep(0.15)
                        keyboard.press_and_release('space')  # Add space after paste
                        paste_success = True
                        print("[PASTE] ✓ Text pasted via clipboard (with trailing space)!")
                    except Exception as e:
                        print(f"[!] Method 3 failed: {e}")
                
                # If all methods failed
                if not paste_success:
                    print("[!] Auto-paste failed with all methods")
                    print("    Text is in clipboard - paste manually with Ctrl+V")
                
                # Play success sound
                threading.Thread(target=self.play_sound, args=(1500, 200), daemon=True).start()
                print("[OK] Success!")
                
            else:
                print("[!] No speech detected")
                # Play error sound
                threading.Thread(target=self.play_sound, args=(400, 300), daemon=True).start()
                
        except Exception as e:
            print(f"[ERROR] Transcription error: {e}")
            import traceback
            traceback.print_exc()
            # Play error sound
            threading.Thread(target=self.play_sound, args=(400, 300), daemon=True).start()
    
    def on_click(self, x, y, button, pressed):
        """Handle mouse button events"""
        if button == mouse.Button.middle:  # Changed to middle button (wheel click)
            if pressed:
                # Middle button pressed - start tracking time
                if not self.is_recording and not self.recording_started:
                    self.press_time = time.time()
            else:
                # Middle button released - check if need to stop recording
                if self.is_recording:
                    self.stop_recording()
                
                # Reset press time and started flag
                self.press_time = None
                self.recording_started = False
    
    def check_hold_duration(self):
        """Check if button has been held long enough to start recording"""
        while True:
            time.sleep(0.05)  # Check more frequently
            if self.press_time and not self.is_recording and not self.recording_started:
                hold_duration = time.time() - self.press_time
                if hold_duration >= 0.2:  # Changed to 0.2 seconds
                    # Mark as started to prevent double-start
                    self.recording_started = True
                    # Start recording in new thread
                    self.recording_thread = threading.Thread(target=self.start_recording)
                    self.recording_thread.start()
    
    def run(self):
        """Start the voice recorder"""
        print("\n" + "="*50)
        print("Voice Recorder with Global Mouse Control")
        print("="*50)
        print("[*] Hold MIDDLE mouse button (wheel) for 0.2+ sec to record")
        print("[*] Release button to stop and transcribe")
        print("[*] Works GLOBALLY - no need to focus this window")
        print("[*] Text will auto-paste to active field!")
        print("[*] Press Ctrl+C to exit")
        print("="*50 + "\n")
        
        # Start thread to check hold duration
        check_thread = threading.Thread(target=self.check_hold_duration, daemon=True)
        check_thread.start()
        
        # Start listening to mouse events (pynput already captures globally)
        print("[OK] Global mouse listener started!")
        with mouse.Listener(on_click=self.on_click) as listener:
            try:
                listener.join()
            except KeyboardInterrupt:
                print("\n\n[EXIT] Exiting...")
                self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if self.is_recording:
            self.stop_recording()
        
        self.audio.terminate()


if __name__ == "__main__":
    # Configuration
    # Model sizes: "tiny", "base", "small", "medium", "large"
    # Smaller models are faster but less accurate
    # Larger models are more accurate but slower
    
    # GPU acceleration (use_cuda=True) significantly speeds up transcription
    # Requires: CUDA-capable GPU and PyTorch with CUDA support
    
    # Language codes: "ru" (Russian), "en" (English), "de" (German), 
    #                 "fr" (French), "es" (Spanish), "it" (Italian),
    #                 "pt" (Portuguese), "pl" (Polish), "uk" (Ukrainian),
    #                 "ja" (Japanese), "zh" (Chinese), "ko" (Korean), etc.
    
    # Microphone gain (mic_gain): Adjusts microphone sensitivity
    #   0.5 = Half volume (quieter, for loud environments)
    #   1.0 = Normal volume (default, recommended)
    #   2.0 = Double volume (louder, for quiet mic or distant speaking)
    #   Range: 0.1 to 5.0
    
    # Playback preview (playback_before_transcribe):
    #   True  = Plays back recording before transcription (for testing/verification)
    #   False = Skip playback, transcribe immediately (default, faster)
    
    # Playback volume (playback_volume): Controls playback loudness
    #   1.0 = Normal (same as recorded)
    #   3.0 = 3x louder (default, recommended for quiet recordings)
    #   5.0 = 5x louder (for very quiet recordings)
    #   10.0 = Maximum (may distort)
    #   Range: 1.0 to 10.0
    #model_size: Whisper model size (tiny, base, small, medium, large)
    recorder = VoiceRecorder(
        model_size="medium",                    # Change to "medium" or "large" for better accuracy
        use_cuda=True,                        # Set to False to force CPU
        language="ru",                        # Set your language: "ru", "en", "de", "fr", etc.
        mic_gain=2.0,                         # Adjust if mic is too quiet (>1.0) or too loud (<1.0)
        playback_before_transcribe=False,      # Set to True to hear recording before transcription
        playback_volume=5.0                   # Playback loudness (1.0-10.0, higher = louder)
    )
    recorder.run()
