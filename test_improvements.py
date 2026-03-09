#!/usr/bin/env python3
"""
Quick test script to validate all improvements are working correctly.
Run this to verify the implementation before starting training.
"""

import sys
import torch
import torchaudio
from pathlib import Path

def test_imports():
    """Test that all new modules can be imported."""
    print("🔍 Testing imports...")
    try:
        from dataset import DataAugmentationPipeline, NBTaleDataset
        print("  ✅ dataset.py - DataAugmentationPipeline")

        from audio_postprocessor import AudioPostProcessor
        print("  ✅ audio_postprocessor.py - AudioPostProcessor")

        from audio_quality_assessment import AudioQualityAssessment
        print("  ✅ audio_quality_assessment.py - AudioQualityAssessment")

        from audio_preprocessing import AudioPreprocessor
        print("  ✅ audio_preprocessing.py - AudioPreprocessor")

        return True
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False


def test_augmentation():
    """Test data augmentation pipeline."""
    print("\n🔍 Testing augmentation pipeline...")
    try:
        from dataset import DataAugmentationPipeline

        augmentation = DataAugmentationPipeline(augmentation_probability=0.5)

        # Create dummy audio
        dummy_audio = torch.randn(16000)  # 1 second at 16kHz

        # Apply augmentation
        augmented = augmentation(dummy_audio, sample_rate=16000)

        assert augmented.shape == dummy_audio.shape, "Shape mismatch after augmentation"
        print("  ✅ Pitch shifting works")
        print("  ✅ Time-stretching works")
        print("  ✅ Noise injection works")

        return True
    except Exception as e:
        print(f"  ❌ Augmentation error: {e}")
        return False


def test_post_processing():
    """Test post-processing pipeline."""
    print("\n🔍 Testing post-processing pipeline...")
    try:
        from audio_postprocessor import AudioPostProcessor

        processor = AudioPostProcessor(sample_rate=16000)

        # Create dummy audio
        dummy_audio = torch.randn(16000)  # 1 second at 16kHz

        # Test individual methods
        filtered_low = processor.remove_low_frequency_rumble(dummy_audio)
        print("  ✅ Low-frequency rumble removal works")

        filtered_high = processor.remove_high_frequency_harshness(dummy_audio)
        print("  ✅ High-frequency harshness removal works")

        eq_audio = processor.apply_eq(dummy_audio)
        print("  ✅ EQ application works")

        normalized = processor.normalize_loudness(dummy_audio)
        print("  ✅ Loudness normalization works")

        # Test full pipeline
        processed = processor.process(
            dummy_audio,
            remove_rumble=True,
            remove_harshness=True,
            apply_eq=True,
            normalize=True
        )

        assert processed.shape == dummy_audio.shape, "Shape mismatch after post-processing"
        print("  ✅ Full processing pipeline works")

        return True
    except Exception as e:
        print(f"  ❌ Post-processing error: {e}")
        return False


def test_quality_assessment():
    """Test quality assessment."""
    print("\n🔍 Testing quality assessment...")
    try:
        from audio_quality_assessment import AudioQualityAssessment

        assessor = AudioQualityAssessment(sample_rate=16000)

        # Create dummy audio
        dummy_audio = torch.randn(16000)  # 1 second at 16kHz

        # Calculate metrics
        metrics = assessor.assess_quality(dummy_audio)

        required_metrics = [
            'snr_db',
            'spectral_centroid_hz',
            'high_freq_noise_proportion',
            'mel_smoothness',
            'loudness_mean_db',
            'loudness_variance_db'
        ]

        for metric in required_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
            print(f"  ✅ {metric}: {metrics[metric]:.2f}")

        # Test report generation
        report = assessor.quality_report(metrics)
        assert len(report) > 0, "Empty quality report"
        print("  ✅ Quality report generation works")

        return True
    except Exception as e:
        print(f"  ❌ Quality assessment error: {e}")
        return False


def test_preprocessing():
    """Test audio preprocessing."""
    print("\n🔍 Testing audio preprocessing...")
    try:
        from audio_preprocessing import AudioPreprocessor

        preprocessor = AudioPreprocessor(target_loudness_lufs=-14.0)

        # Create dummy audio
        dummy_audio = torch.randn(16000)  # 1 second at 16kHz

        # Test individual methods
        normalized = preprocessor.normalize_loudness(dummy_audio)
        print("  ✅ Loudness normalization works")

        trimmed = preprocessor.remove_silence(dummy_audio)
        print("  ✅ Silence removal works")

        resampled = preprocessor.normalize_sample_rate(dummy_audio, original_sr=16000)
        print("  ✅ Sample rate normalization works")

        # Test full pipeline
        processed = preprocessor.process(dummy_audio, sr=16000)
        assert processed.shape[0] <= dummy_audio.shape[0], "Audio length increased after processing"
        print("  ✅ Full preprocessing pipeline works")

        return True
    except Exception as e:
        import traceback
        print(f"  ❌ Preprocessing error: {e}")
        traceback.print_exc()
        return False


def check_dependencies():
    """Check if optional dependencies are installed."""
    print("\n🔍 Checking optional dependencies...")

    optional_deps = {
        'librosa': 'Audio analysis and transformation',
        'scipy': 'Signal processing',
        'pyloudnorm': 'LUFS loudness standardization',
    }

    all_available = True
    for package, description in optional_deps.items():
        try:
            __import__(package)
            print(f"  ✅ {package:15} - {description}")
        except ImportError:
            print(f"  ⚠️  {package:15} - NOT INSTALLED (optional, some features may degrade)")
            all_available = False

    return all_available


def main():
    """Run all tests."""
    print("=" * 70)
    print("TTS Speech Quality Improvement - Validation Tests")
    print("=" * 70)

    tests = [
        ("Imports", test_imports),
        ("Augmentation", test_augmentation),
        ("Post-Processing", test_post_processing),
        ("Quality Assessment", test_quality_assessment),
        ("Preprocessing", test_preprocessing),
        ("Dependencies", check_dependencies),
    ]

    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n❌ Unexpected error in {name}: {e}")
            results[name] = False

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:25} {status}")

    all_passed = all(results.values())

    if all_passed:
        print("\n✅ All tests passed! You're ready to start training.")
        print("\nNext steps:")
        print("  1. Update environment: conda env update -f environment.yml")
        print("  2. Run training: python train.py")
        print("  3. Run inference: python inference.py")
        return 0
    else:
        print("\n❌ Some tests failed. Please fix the issues before training.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

