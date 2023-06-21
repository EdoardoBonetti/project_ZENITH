import asyncio
from ffmpeg import FFmpeg

ffmpeg = FFmpeg().option('y').input(
    'rtsp://example.com/cam',
    # Specify file options using kwargs
    rtsp_transport='tcp',
    rtsp_flags='prefer_tcp',
).output(
    'output.ts',
    # Use a dictionary when an option name contains special characters
    {'codec:v': 'copy'},
    f='mpegts',
)

@ffmpeg.on('start')
def on_start(arguments):
    print('Arguments:', arguments)

@ffmpeg.on('stderr')
def on_stderr(line):
    print('stderr:', line)

@ffmpeg.on('progress')
def on_progress(progress):
    print(progress)

@ffmpeg.on('progress')
def time_to_terminate(progress):
    # Gracefully terminate when more than 200 frames are processed
    if progress.frame > 200:
        ffmpeg.terminate()

@ffmpeg.on('completed')
def on_completed():
    print('Completed')

@ffmpeg.on('terminated')
def on_terminated():
    print('Terminated')

@ffmpeg.on('error')
def on_error(code):
    print('Error:', code)

asyncio.run(ffmpeg.execute())