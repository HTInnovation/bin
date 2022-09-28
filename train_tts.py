import os

from trainer import Trainer, TrainerArgs
from TTS.tts.configs.glow_tts_config import GlowTTSConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.glow_tts import GlowTTS
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

output_path = os.path.dirname(os.path.abspath(__file__))

dataset_config = BaseDatasetConfig(
    dataset_name="vctk", meta_file_train="meta.csv", path="data/book/", language="en-us"
)

def data_formatter(root_path, manifest_file, **kwargs):
    speaker_name = "my_speaker"
    items = list()

    with open(os.path.join(root_path, manifest_file), "r", encoding="utf-8") as stream:
        lines = stream.read().strip().splitlines()
        for line in lines:
            tg = line.split("\t")
            f_name = tg[0].split("_")
            f_name = os.path.join(root_path, "auds", f"{f_name[0]}_{int(f_name[1]):06d}.wav")
            
            if os.path.exists(f_name) is True:
                items.append({
                    "text": tg[1].strip(),
                    "audio_file": f_name,
                    "speaker_name": speaker_name,
                    "root_path": root_path
                })

    return items

config = GlowTTSConfig(
    batch_size=32,
    eval_batch_size=16,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    text_cleaner="phoneme_cleaners",
    use_phonemes=True,
    phoneme_language="vi",
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    print_step=25,
    print_eval=False,
    mixed_precision=True,
    output_path=output_path,
    datasets=[dataset_config],
)

ap = AudioProcessor.init_from_config(config)

tokenizer, config = TTSTokenizer.init_from_config(config)

train_samples, eval_samples = load_tts_samples(
    [dataset_config],
    eval_split=True,
    formatter=data_formatter
)

model = GlowTTS(config, ap, tokenizer, speaker_manager=None)

trainer = Trainer(
    TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
)

trainer.fit()
