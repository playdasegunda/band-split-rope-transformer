import numpy as np
import ffmpeg
import librosa

def ffmpeg_processor(input_file):
    all_channels = []

    probe = ffmpeg.probe(input_file)
    audio_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'audio']
    indices = len(audio_streams)
    channels = audio_streams[0]['channels']

    for index in range(indices):
        for channel in range(channels):
            try:
                out, _ = (
                        ffmpeg
                        .input(input_file)
                        .output('pipe:', format='f32le', ac=1, filter=f'pan=mono|c0=c{channel}', map=f'0:a:{index}')
                        .run(capture_stdout=True, capture_stderr=True)
                    )
            except ffmpeg.Error as e:
                print(e.stderr.decode())
                raise

            audio_array = np.frombuffer(out, np.float32)
            all_channels.append(audio_array)

    combined_channels = np.stack(all_channels, axis=0)
    return combined_channels

def flac_processor(input_file):
    out, sample_rate = librosa.load(input_file, sr=None)
    return out

if __name__ == '__main__':
    out = ffmpeg_processor('../A Classic Education - NightOwl.stem.mp4')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(len(out))
    for index, data in enumerate(out):
        axs[index].plot(out[index])
        axs[index].set_ylim(-1, 1)
    plt.show()