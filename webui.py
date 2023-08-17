import gradio as gr
from glob import glob
from text import text_to_sequence
import commons
import torch
import utils
import sys
import os
from text.symbols import symbols
from models import SynthesizerTrn
from scipy.io.wavfile import write

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)

    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    
    text_norm = torch.LongTensor(text_norm)
    return text_norm


def infer_button_click(infer_text, pth_file_path, model_name):
    hps = utils.get_hparams_from_file(f"models/{model_name}/config.json")
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model
    ).cuda()

    _ = utils.load_checkpoint(pth_file_path, net_g, None)
    _ = net_g.eval()

    sid = torch.LongTensor([0]).cuda()
    stn_tst = get_text(infer_text, hps)
    with torch.no_grad():
        x_tst = stn_tst.cuda().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
        audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale= 1 / 1)[0][0, 0].data.cpu().float().numpy()
        audio_file_name = f"output/{model_name}/output_0.wav"
        write(audio_file_name, hps.data.sampling_rate, audio)
        return audio_file_name

def gradio_app(theme=gr.themes.Soft()):

    with gr.Blocks(theme=theme, title="hellow") as app:
        
        with gr.Row():
            pth_file_path = gr.Textbox(
                label="Pth file path"
            )
        
            models = gr.Dropdown(
                label="Select model",
                choices=[os.path.basename(dirs) for dirs in glob("models/*")]
            )

        with gr.Row():
            infer_text = gr.Textbox(
                label="Inperence Text"
            )

            infer_button = gr.Button(
                value="Submit",
                variant="secondary",
            )

        with gr.Row():
            make_auido = gr.Audio(
                source="upload",
                type="filepath"
            )
        

        

        infer_button.click(
            fn=infer_button_click,
            inputs=[infer_text, pth_file_path, models],
            outputs=[make_auido]
        )
        
        return app
    
def gradio_run(app):
    app.queue(
        concurrency_count=511,
        max_size=1022,
    ).launch(
        server_name="0.0.0.0",
        inbrowser=False,
        quiet=True,
    )


if __name__ == "__main__":
    app = gradio_app()
    gradio_run(app)