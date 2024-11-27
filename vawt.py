# vawt.py

# packages to install
# torch whisper pyannote.audio

import argparse
import torch
import whisper
from pyannote.audio import Pipeline
from subprocess import CalledProcessError, run

HUGGING_FACE_ACCESS_TOKEN='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'

def to_wav(file, to):
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", file,
        "-f", "s16le",
        "-ac", "1",
        "-ar", 16000,
        "-c:a", "pcm_s16le",
        "-af", "silenceremove=1:0:-50dB",
        to
    ]
    try:
        run(cmd, check=True)
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

def do_it(audio_name, srt_name, model_name="medium", hugging_face_access_token=HUGGING_FACE_ACCESS_TOKEN):
    # devices: cpu, cuda or mps
    # if torch.backends.cuda.is_avaialble():
    #     device = torch.device("cuda")
    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")
    # else:
    #     device = torch.device("cpu")

    device = torch.device("cuda")
    model = whisper.load_model(model_name)
    model.to(device)

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", 
        use_auth_token=hugging_face_access_token)
    pipeline.to(device)

    audio = whisper.load_audio(audio_name)
    result = model.transcribe(audio=audio, temperature=0.1, verbose=False, word_timestamps=True)
    diarization = pipeline(audio_name)

    writer = whisper.utils.WriteSRT(".")
    with open(srt_name, "w", encoding="utf-8") as f: 
        writer.write_result(result, file=f)

def main():
    parser = argparse.ArgumentParser(prog='vawt')

    subparsers = parser.add_subparsers(dest='command', required=True)

    # 'to-wav' command
    parser_to_wav = subparsers.add_parser('to-wav', help='Convert input file to WAV format')
    parser_to_wav.add_argument('in_file', help='Input audio file')
    parser_to_wav.add_argument('out_file', help='Output WAV file')
    # parser_to_wav.add_argument('--option1', help='Option 1 description')

    # 'transcribe' command
    parser_transcribe = subparsers.add_parser('transcribe', help='Transcribe audio file')
    parser_transcribe.add_argument('in_file', help='Input audio file')
    parser_transcribe.add_argument('srt_file', help='Output srt file')
    # parser_transcribe.add_argument('--option2', help='Option 2 description')

    args = parser.parse_args()

    if args.command == 'to-wav':
        # Handle 'to-wav' command
        in_file = args.in_file
        out_file = args.out_file
        option1 = args.option1
        to_wav(in_file, out_file)
    elif args.command == 'transcribe':
        # Handle 'transcribe' command
        in_file = args.in_file
        out_file = args.srt_file
        do_it(in_file, out_file, model_name="medium", hugging_face_access_token=HUGGING_FACE_ACCESS_TOKEN)

if __name__ == '__main__':
    main()

# EOF
