import argparse
import platform
from dataclasses import dataclass
from datetime import datetime

from audiocraft.data.audio import audio_write
from audiocraft.models import MusicGen


@dataclass
class Prompt:
    description: str
    duration: int
    model: str


available_models: dict[str, MusicGen] = {}


def generate_music(prompt: Prompt):
    # this logic is for supporting multiple prompts sequentially in the same process
    if prompt.model in available_models:
        print(f'retrieving existing {prompt.model} model instance')
        model = available_models[prompt.model]
    else:
        print(f'fetching new instance of {prompt.model} model')
        model = MusicGen.get_pretrained(prompt.model)
        available_models[prompt.model] = model

    model.set_generation_params(duration=prompt.duration)

    print(f'generating music for "{prompt.description}"')
    start_time = datetime.now()
    generated_music = model.generate([prompt.description], progress=True)[0]
    end_time = datetime.now()
    generation_time = end_time - start_time
    generation_summary = (f'~~~~ generated {prompt.duration} seconds of music in {generation_time} '
                          f'using model {prompt.model} on machine {platform.node()} ~~~~')
    print(generation_summary)
    with open('generation-times.txt', 'a') as f:
        f.write(f'{generation_summary}\n')

    filename = f'generated/[{datetime.now()}] {prompt.description}'
    print(f'writing audio to {filename}.wav')
    audio_write(filename, generated_music.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--description",
                        help="prompt description of music to generate")
    parser.add_argument("-D", "--duration",
                        help="duration in seconds of generated audio",
                        type=int, required=False, default=10)
    parser.add_argument("-m", "--model",
                        help="name of model to use (facebook/musicgen-small, facebook/musicgen-melody, "
                             "facebook/musicgen-large, facebook/musicgen-stereo-small, etc)",
                        required=False, default='facebook/musicgen-small')
    args = parser.parse_args()
    generate_music(Prompt(args.description, args.duration, args.model))
