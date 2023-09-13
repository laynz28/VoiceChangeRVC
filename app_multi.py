from typing import Union

from argparse import ArgumentParser

import asyncio
import json
import hashlib
from os import path, getenv

import gradio as gr

import torch

import numpy as np
import librosa

import edge_tts

import config
import util
from fairseq import checkpoint_utils
from infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from vc_infer_pipeline import VC
from config import Config
config = Config()
force_support = None
if config.unsupported is False:
    if config.device == "mps" or config.device == "cpu":
        force_support = False
else:
    force_support = True
    
# Reference: https://huggingface.co/spaces/zomehwh/rvc-models/blob/main/app.py#L21  # noqa
in_hf_space = getenv('SYSTEM') == 'spaces'

# Argument parsing
arg_parser = ArgumentParser()
arg_parser.add_argument(
    '--hubert',
    default=getenv('RVC_HUBERT', 'hubert_base.pt'),
    help='path to hubert base model (default: hubert_base.pt)'
)
arg_parser.add_argument(
    '--config',
    default=getenv('RVC_MULTI_CFG', 'multi_config.json'),
    help='path to config file (default: multi_config.json)'
)
arg_parser.add_argument(
    '--api',
    action='store_true',
    help='enable api endpoint'
)
arg_parser.add_argument(
    '--cache-examples',
    action='store_true',
    help='enable example caching, please remember delete gradio_cached_examples folder when example config has been modified'  # noqa
)
args = arg_parser.parse_args()

app_css = '''
#model_info img {
    max-width: 100px;
    max-height: 100px;
    float: right;
}

#model_info p {
    margin: unset;
}
'''

app = gr.Blocks(
    theme=gr.themes.Soft(primary_hue="orange", secondary_hue="slate"),
    css=app_css,
    analytics_enabled=False
)

# Load hubert model
models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
    ["hubert_base.pt"],
    suffix="",
)
hubert_model = models[0]
hubert_model = hubert_model.to(config.device)
if config.is_half:
    hubert_model = hubert_model.half()
else:
    hubert_model = hubert_model.float()
hubert_model.eval()

# Load models
multi_cfg = json.load(open(args.config, 'r'))
loaded_models = []

for model_name in multi_cfg.get('models'):
    print(f'Loading model: {model_name}')

    # Load model info
    model_info = json.load(
        open(path.join('model', model_name, 'config.json'), 'r')
    )

    # Load RVC checkpoint
    cpt = torch.load(
        path.join('model', model_name, model_info['model']),
        map_location='cpu'
    )
    tgt_sr = cpt['config'][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")
    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
        model_version = "V1"
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
        model_version = "V2"
    del net_g.enc_q

    print(net_g.load_state_dict(cpt["weight"], strict=False))
    net_g.eval().to(config.device)
    if config.is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()
    vc = VC(tgt_sr, config)
    
    loaded_models.append(dict(
        name=model_name,
        metadata=model_info,
        vc=vc,
        net_g=net_g,
        if_f0=if_f0,
        target_sr=tgt_sr,
        test=model_version
    ))
        
print(f'Models loaded: {len(loaded_models)}')

# Edge TTS speakers
tts_speakers_list = asyncio.get_event_loop().run_until_complete(edge_tts.list_voices())  # noqa


# https://github.com/fumiama/Retrieval-based-Voice-Conversion-WebUI/blob/main/infer-web.py#L118  # noqa
def vc_func(
    input_audio, model_index, pitch_adjust, f0_method, feat_ratio,
    filter_radius, rms_mix_rate, resample_option
):
    if input_audio is None:
        return (None, 'Please provide input audio.')

    if model_index is None:
        return (None, 'Please select a model.')

    model = loaded_models[model_index]

    # Reference: so-vits
    (audio_samp, audio_npy) = input_audio

    # https://huggingface.co/spaces/zomehwh/rvc-models/blob/main/app.py#L49
    # Can be change well, we will see
    if (audio_npy.shape[0] / audio_samp) > 320 and in_hf_space:
        return (None, 'Input audio is longer than 60 secs.')

    # Bloody hell: https://stackoverflow.com/questions/26921836/
    if audio_npy.dtype != np.float32:  # :thonk:
        audio_npy = (
            audio_npy / np.iinfo(audio_npy.dtype).max
        ).astype(np.float32)

    if len(audio_npy.shape) > 1:
        audio_npy = librosa.to_mono(audio_npy.transpose(1, 0))

    if audio_samp != 16000:
        audio_npy = librosa.resample(
            audio_npy,
            orig_sr=audio_samp,
            target_sr=16000
        )

    pitch_int = int(pitch_adjust)

    resample = (
        0 if resample_option == 'Disable resampling'
        else int(resample_option)
    )

    times = [0, 0, 0]

    checksum = hashlib.sha512()
    checksum.update(audio_npy.tobytes())

    print(model['test'])
    
    output_audio = model['vc'].pipeline(
        hubert_model,
        model['net_g'],
        model['metadata'].get('speaker_id', 0),
        audio_npy,
        checksum.hexdigest(),
        times,
        pitch_int,
        f0_method,
        path.join('model', model['name'], model['metadata']['feat_index']),
        feat_ratio,
        model['if_f0'],
        filter_radius,
        model['target_sr'],
        resample,
        rms_mix_rate,
        model['test']
        0.5    
    )

    out_sr = (
        resample if resample >= 16000 and model['target_sr'] != resample
        else model['target_sr']
    )

    print(f'npy: {times[0]}s, f0: {times[1]}s, infer: {times[2]}s')
    return ((out_sr, output_audio), 'Success')


async def edge_tts_vc_func(
    input_text, model_index, tts_speaker, pitch_adjust, f0_method, feat_ratio,
    filter_radius, rms_mix_rate, resample_option
):
    if input_text is None:
        return (None, 'Please provide TTS text.')

    if tts_speaker is None:
        return (None, 'Please select TTS speaker.')

    if model_index is None:
        return (None, 'Please select a model.')

    speaker = tts_speakers_list[tts_speaker]['ShortName']
    (tts_np, tts_sr) = await util.call_edge_tts(speaker, input_text)
    return vc_func(
        (tts_sr, tts_np),
        model_index,
        pitch_adjust,
        f0_method,
        feat_ratio,
        filter_radius,
        rms_mix_rate,
        resample_option
    )


def update_model_info(model_index):
    if model_index is None:
        return str(
            '### Model info\n'
            'Please select a model from dropdown above.'
        )

    model = loaded_models[model_index]
    model_icon = model['metadata'].get('icon', '')

    return str(
        '### Model info\n'
        '![model icon]({icon})'
        '**{name}**\n\n'
        'Author: {author}\n\n'
        'Source: {source}\n\n'
        '{note}'
    ).format(
        name=model['metadata'].get('name'),
        author=model['metadata'].get('author', 'Anonymous'),
        source=model['metadata'].get('source', 'Unknown'),
        note=model['metadata'].get('note', ''),
        icon=(
            model_icon
            if model_icon.startswith(('http://', 'https://'))
            else '/file/model/%s/%s' % (model['name'], model_icon)
        )
    )


def _example_vc(
    input_audio, model_index, pitch_adjust, f0_method, feat_ratio,
    filter_radius, rms_mix_rate, resample_option
):
    (audio, message) = vc_func(
        input_audio, model_index, pitch_adjust, f0_method, feat_ratio,
        filter_radius, rms_mix_rate, resample_option
    )
    return (
        audio,
        message,
        update_model_info(model_index)
    )


async def _example_edge_tts(
    input_text, model_index, tts_speaker, pitch_adjust, f0_method, feat_ratio,
    filter_radius, rms_mix_rate, resample_option
):
    (audio, message) = await edge_tts_vc_func(
        input_text, model_index, tts_speaker, pitch_adjust, f0_method,
        feat_ratio, filter_radius, rms_mix_rate, resample_option
    )
    return (
        audio,
        message,
        update_model_info(model_index)
    )


with app:
    gr.Markdown(
        '## A simplistic Web interface\n'
        'RVC interface, project based on [RVC-WebUI](https://github.com/fumiama/Retrieval-based-Voice-Conversion-WebUI)'  # thx noqa
        'A lot of inspiration from what\'s already out there, including [zomehwh/rvc-models](https://huggingface.co/spaces/zomehwh/rvc-models) & [DJQmUKV/rvc-inference](https://huggingface.co/spaces/DJQmUKV/rvc-inference).\n '  # thx noqa
    )

    with gr.Row():
        with gr.Column():
            with gr.Tab('Audio conversion'):
                input_audio = gr.Audio(label='Input audio')

                vc_convert_btn = gr.Button('Convert', variant='primary')

            with gr.Tab('TTS conversion'):
                tts_input = gr.TextArea(
                    label='TTS input text'
                )
                tts_speaker = gr.Dropdown(
                    [
                        '%s (%s)' % (
                            s['FriendlyName'],
                            s['Gender']
                        )
                        for s in tts_speakers_list
                    ],
                    label='TTS speaker',
                    type='index'
                )

                tts_convert_btn = gr.Button('Convert', variant='primary')

            pitch_adjust = gr.Slider(
                label='Pitch',
                minimum=-24,
                maximum=24,
                step=1,
                value=0
            )
            f0_method = gr.Radio(
                label='f0 methods',
                choices=['pm', 'harvest', 'crepe'],
                value='pm',
                interactive=True
            )

            with gr.Accordion('Advanced options', open=False):
                feat_ratio = gr.Slider(
                    label='Feature ratio',
                    minimum=0,
                    maximum=1,
                    step=0.1,
                    value=0.6
                )
                filter_radius = gr.Slider(
                    label='Filter radius',
                    minimum=0,
                    maximum=7,
                    step=1,
                    value=3
                )
                rms_mix_rate = gr.Slider(
                    label='Volume envelope mix rate',
                    minimum=0,
                    maximum=1,
                    step=0.1,
                    value=1
                )
                resample_rate = gr.Dropdown(
                    [
                        'Disable resampling',
                        '16000',
                        '22050',
                        '44100',
                        '48000'
                    ],
                    label='Resample rate',
                    value='Disable resampling'
                )

        with gr.Column():
            # Model select
            model_index = gr.Dropdown(
                [
                    '%s - %s' % (
                        m['metadata'].get('source', 'Unknown'),
                        m['metadata'].get('name')
                    )
                    for m in loaded_models
                ],
                label='Model',
                type='index'
            )

            # Model info
            with gr.Box():
                model_info = gr.Markdown(
                    '### Model info\n'
                    'Please select a model from dropdown above.',
                    elem_id='model_info'
                )

            output_audio = gr.Audio(label='Output audio')
            output_msg = gr.Textbox(label='Output message')

    multi_examples = multi_cfg.get('examples')
    if (
        multi_examples and
        multi_examples.get('vc') and multi_examples.get('tts_vc')
    ):
        with gr.Accordion('Sweet sweet examples', open=False):
            with gr.Row():
                # VC Example
                if multi_examples.get('vc'):
                    gr.Examples(
                        label='Audio conversion examples',
                        examples=multi_examples.get('vc'),
                        inputs=[
                            input_audio, model_index, pitch_adjust, f0_method,
                            feat_ratio
                        ],
                        outputs=[output_audio, output_msg, model_info],
                        fn=_example_vc,
                        cache_examples=args.cache_examples,
                        run_on_click=args.cache_examples
                    )

                # Edge TTS Example
                if multi_examples.get('tts_vc'):
                    gr.Examples(
                        label='TTS conversion examples',
                        examples=multi_examples.get('tts_vc'),
                        inputs=[
                            tts_input, model_index, tts_speaker, pitch_adjust,
                            f0_method, feat_ratio
                        ],
                        outputs=[output_audio, output_msg, model_info],
                        fn=_example_edge_tts,
                        cache_examples=args.cache_examples,
                        run_on_click=args.cache_examples
                    )

    vc_convert_btn.click(
        vc_func,
        [
            input_audio, model_index, pitch_adjust, f0_method, feat_ratio,
            filter_radius, rms_mix_rate, resample_rate
        ],
        [output_audio, output_msg],
        api_name='audio_conversion'
    )

    tts_convert_btn.click(
        edge_tts_vc_func,
        [
            tts_input, model_index, tts_speaker, pitch_adjust, f0_method,
            feat_ratio, filter_radius, rms_mix_rate, resample_rate
        ],
        [output_audio, output_msg],
        api_name='tts_conversion'
    )

    model_index.change(
        update_model_info,
        inputs=[model_index],
        outputs=[model_info],
        show_progress=False,
        queue=False
    )

app.queue(
    concurrency_count=1,
    max_size=20,
    api_open=args.api
).launch()