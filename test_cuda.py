#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CUDA Availability Test
Tests if CUDA is available and provides GPU information
"""

import sys

def test_cuda():
    """Test CUDA availability and show GPU information"""
    print("=" * 60)
    print("            CUDA AVAILABILITY TEST")
    print("=" * 60)
    print()
    
    # Test PyTorch
    print("📦 Testing PyTorch...")
    try:
        import torch
        print("✅ PyTorch is installed")
        print(f"   Version: {torch.__version__}")
        print()
    except ImportError:
        print("❌ PyTorch is NOT installed")
        print("   Install: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print()
        return False
    
    # Test CUDA availability
    print("🚀 Testing CUDA...")
    cuda_available = torch.cuda.is_available()
    
    if cuda_available:
        print("✅ CUDA is AVAILABLE! 🎉")
        print()
        
        # GPU information
        gpu_count = torch.cuda.device_count()
        print(f"🎮 Number of GPUs: {gpu_count}")
        print()
        
        for i in range(gpu_count):
            print(f"GPU {i}:")
            print(f"   Name: {torch.cuda.get_device_name(i)}")
            
            # GPU memory
            props = torch.cuda.get_device_properties(i)
            total_memory = props.total_memory / (1024**3)  # Convert to GB
            print(f"   Total Memory: {total_memory:.2f} GB")
            print(f"   Compute Capability: {props.major}.{props.minor}")
            
            # Current memory usage
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                reserved = torch.cuda.memory_reserved(i) / (1024**3)
                print(f"   Memory Allocated: {allocated:.2f} GB")
                print(f"   Memory Reserved: {reserved:.2f} GB")
            print()
        
        # CUDA version
        print(f"🔧 CUDA Version: {torch.version.cuda}")
        print(f"🔧 cuDNN Version: {torch.backends.cudnn.version()}")
        print()
        
        # Test tensor operation on GPU
        print("🧪 Testing GPU tensor operation...")
        try:
            x = torch.rand(1000, 1000).cuda()
            y = torch.rand(1000, 1000).cuda()
            z = torch.matmul(x, y)
            print("✅ GPU tensor operations work correctly!")
            print()
        except Exception as e:
            print(f"❌ GPU tensor operation failed: {e}")
            print()
            return False
        
        # Recommendation
        print("=" * 60)
        print("✨ RECOMMENDATION")
        print("=" * 60)
        print("Your GPU is ready for acceleration!")
        print("In SpeechToText.py, set:")
        print("    use_cuda=True")
        print()
        print("This will make speech recognition 5-10x faster! 🚀")
        print("=" * 60)
        
        return True
        
    else:
        print("❌ CUDA is NOT available")
        print()
        print("Possible reasons:")
        print("1. No NVIDIA GPU in your system")
        print("2. GPU drivers are not installed")
        print("3. PyTorch was installed without CUDA support")
        print()
        print("To install PyTorch with CUDA:")
        print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print()
        print("=" * 60)
        print("⚠️  RECOMMENDATION")
        print("=" * 60)
        print("CUDA is not available. The program will use CPU.")
        print("In SpeechToText.py, set:")
        print("    use_cuda=False")
        print()
        print("CPU mode works but is slower (5-10x).")
        print("=" * 60)
        
        return False


if __name__ == "__main__":
    try:
        success = test_cuda()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
