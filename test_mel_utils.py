import sys
# Add the directory of whisper_mel_utils.py to sys.path
# This is to allow Python to find the tts.utils.audio_utils module
# Assumes this script is run from the root of the repo
sys.path.insert(0, ".")

try:
    from tts.utils.audio_utils.whisper_mel_utils import mel_filters
    # Attempt to call mel_filters. This should try to create the 'assets' directory.
    # We expect a FileNotFoundError because 'mel_filters.npz' is not present.
    # We pass a dummy device and n_mels value.
    mel_filters(device="cpu", n_mels=80)
except FileNotFoundError as e:
    print(f"Successfully caught expected error: {e}")
    # Check if the assets directory was created by inspecting the error message or by a separate check
    if "Created directory" in str(e) or "assets' directory could not be created" not in str(e):
        # This part of the message "Please ensure 'mel_filters.npz' is placed in tts/utils/audio_utils/assets"
        # is printed by the script itself if it creates the directory.
        # We can also check the specific error message that tells us the file is missing (not the dir)
        if "mel_filters.npz' not found" in str(e):
             print("Directory 'tts/utils/audio_utils/assets/' was likely created by the script.")
        else:
            # This case might indicate the directory was not created, or some other FileNotFoundError happened.
            print(f"Directory 'tts/utils/audio_utils/assets/' may not have been created or another issue occurred. Error: {e}")
    else:
        print(f"Directory 'tts/utils/audio_utils/assets/' was NOT created. Error: {e}")

except Exception as e:
    print(f"An unexpected error occurred: {e}")
    sys.exit(1) # Exit with error
