import whisper
import os
import torch
import datetime

def transcribe_audio(model_size='small', language='slovak'):
    """
    Transcribes an audio file to text using OpenAI Whisper.

    :param model_size: Model size (tiny, base, small, medium, large).
    :param language: Language of the transcription (default: Slovak).
    :return: Transcribed text.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, "models")  # Custom model directory
    os.environ["XDG_CACHE_HOME"] = models_dir  # Set cache to models directory

    audio_path = os.path.join(script_dir, "audio.mp3")
    output_dir = os.path.join(script_dir, "transcripts")  # Directory for output files

    if not os.path.exists(audio_path):
        print(f"File {audio_path} does not exist.")
        return

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Ensure models directory exists
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # Generate timestamp for unique filenames
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    transcript_file = os.path.join(output_dir, f"audio_transcript_{timestamp}.txt")
    log_file = os.path.join(output_dir, f"transcription_log_{timestamp}.log")

    # Redirect console output to log file
    with open(log_file, "w", encoding="utf-8") as log:
        log.write(f"Transcription started at {timestamp}\n")
        log.write(f"Model: {model_size}, Language: {language}\n")

        # Use CPU instead of MPS (MPS is unstable for Whisper)
        device = "cpu"
        log.write("Using CPU (MPS is not stable for Whisper on macOS).\n")

        # Check if model exists in the custom cache directory
        model_path = os.path.join(models_dir, "whisper", model_size + ".pt")
        if os.path.exists(model_path):
            print(f"Using cached model: {model_path}")
            log.write(f"Using cached model: {model_path}\n")
        else:
            print(f"Downloading model: {model_size} to {models_dir}")
            log.write(f"Downloading model: {model_size} to {models_dir}\n")

        print(f"Loading Whisper model: {model_size} on {device} ...")
        model = whisper.load_model(model_size, device=device)

        # Optimize CPU usage
        torch.set_num_threads(4)

        log.write(f"Processing audio file: {audio_path}\n")
        result = model.transcribe(audio_path, language=language, fp16=False)  # Disable FP16

        transcript = result['text']
        log.write("Transcription completed.\n")

        # Save transcription to a timestamped file
        with open(transcript_file, "w", encoding="utf-8") as f:
            f.write(transcript)

        print(f"Transcription saved to: {transcript_file}")
        log.write(f"Transcription saved to: {transcript_file}\n")

    print(f"Log saved to: {log_file}")

if __name__ == "__main__":
    model_size = input("Enter model size (tiny, base, small, medium, large, default: small): ") or 'small'
    language = input("Enter language for transcription (default: Slovak, examples: English -> 'english', German -> 'german'): ") or 'slovak'
    transcribe_audio(model_size, language)