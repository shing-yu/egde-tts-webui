# app.py
import asyncio
import os
import re
from pydub import AudioSegment
import edge_tts
import time
import srt
import uuid
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session, jsonify, copy_current_request_context
from edge_tts.typing import TTSChunk

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # 替换为你的密钥
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'
app.config['MAX_CONTENT_LENGTH'] = 30 * 1024 * 1024  # 限制上传为30MB

# 确保目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# 语音配置
VOICES = [
    {"name": "晓晓", "id": "zh-CN-XiaoxiaoNeural", "gender": "女",
     "content": ["新闻", "小说"], "style": "温暖"},
    {"name": "晓伊", "id": "zh-CN-XiaoyiNeural", "gender": "女",
     "content": ["卡通", "小说"], "style": "活泼"},
    {"name": "云健", "id": "zh-CN-YunjianNeural", "gender": "男",
     "content": ["体育", "小说"], "style": "激情"},
    {"name": "云希", "id": "zh-CN-YunxiNeural", "gender": "男",
     "content": ["小说"], "style": "活泼, 阳光"},
    {"name": "云夏", "id": "zh-CN-YunxiaNeural", "gender": "男",
     "content": ["卡通", "小说"], "style": "可爱"},
    {"name": "云扬", "id": "zh-CN-YunyangNeural", "gender": "男",
     "content": ["新闻"], "style": "专业, 可靠"},
    {"name": "辽宁晓北", "id": "zh-CN-liaoning-XiaobeiNeural", "gender": "女",
     "content": ["方言"], "style": "幽默"},
    {"name": "陕西晓妮", "id": "zh-CN-shaanxi-XiaoniNeural", "gender": "女",
     "content": ["方言"], "style": "明亮"},
]

# 文本分割的近似字符数
CHUNK_SIZE = 3000
# 预计每千字耗时（秒）
ESTIMATED_TIME_PER_1000_CHARS = 8.5

# 存储转换状态的全局字典
conversion_status = {}


class SubMakerPlus(edge_tts.SubMaker):
    """
    A subclass of edge_tts.SubMaker that supports merging multiple audios into a single SRT file.
    """
    def __init__(self):
        super().__init__()

    def feed(self, msg: TTSChunk, offset: int = 0) -> None:
        """
        Feed a WordBoundary message to the SubMaker object.

        Args:
            msg (dict): The WordBoundary message.
            offset (int): An optional offset to adjust the start time of the subtitle.

        Returns:
            None
        """
        if msg["type"] != "WordBoundary":
            raise ValueError("Invalid message type, expected 'WordBoundary'")
        
        msg["offset"] += offset  # Adjust offset if needed

        self.cues.append(
            srt.Subtitle(
                index=len(self.cues) + 1,
                start=srt.timedelta(microseconds=msg["offset"] / 10),
                end=srt.timedelta(microseconds=(msg["offset"] + msg["duration"]) / 10),
                content=msg["text"],
            )
        )

# --- 功能实现 ---

def split_text(text, max_len):
    """
    将文本按标点符号分割成长度不超过 max_len 的块。
    """
    # 使用多种标点符号进行分割
    delimiters = "。？！"
    # 正则表达式，确保在分割后保留标点符号
    pattern = f"([{delimiters}])"

    sentences = re.split(pattern, text)

    # 将句子和其后的标点合并
    sentences = ["".join(x) for x in zip(sentences[0::2], sentences[1::2])]

    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_len:
            current_chunk += sentence
        else:
            chunks.append(current_chunk)
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


async def text_to_speech_chunks(chunks, voice, output_filename, srt_filename, 
                                rate="+0%", volume="+0%", pitch="+0Hz", 
                                words_in_cue=10, task_id=None):
    """
    将文本块列表转换为多个 MP3 音频文件，并生成SRT字幕。
    """
    global conversion_status
    
    temp_files = []
    sub_offset = 0  # Offset for subtitles in each chunk
    submaker = SubMakerPlus()
    total_chunks = len(chunks)
    
    for i, chunk in enumerate(chunks):
        temp_filename = os.path.join(app.config['OUTPUT_FOLDER'], f"temp_chunk_{i}.mp3")
        temp_files.append(temp_filename)
        
        # 更新状态
        progress = int((i / total_chunks) * 80)  # 80%用于音频生成
        conversion_status[task_id] = {
            'status': 'processing',
            'progress': progress,
            'message': f"正在生成第 {i+1}/{total_chunks} 个音频片段..."
        }

        start_time = time.time()
        print(f"正在生成第 {i+1}/{total_chunks} 个音频片段...")
        communicate = edge_tts.Communicate(chunk, voice, rate=rate, volume=volume, pitch=pitch)
        with open(temp_filename, "wb") as file:
            async for chunk_ in communicate.stream():
                if chunk_["type"] == "audio":
                    file.write(chunk_["data"])
                elif chunk_["type"] == "WordBoundary":
                    submaker.feed(chunk_, offset=sub_offset)

        end_time = time.time()
        elapsed_time = end_time - start_time

        # 音频时长
        audio_duration = AudioSegment.from_mp3(temp_filename).duration_seconds
        print(f"片段 {i+1} 生成完成，耗时: {elapsed_time:.2f} 秒，音频时长: {audio_duration:.2f} 秒")
        sub_offset += audio_duration * 10**7
    
    # 更新合并状态
    conversion_status[task_id] = {
        'status': 'processing',
        'progress': 85,
        'message': "音频生成完成，正在合并文件..."
    }
    
    # 合并音频文件
    print("正在合并所有音频片段...")
    combined = AudioSegment.empty()
    for file in temp_files:
        sound = AudioSegment.from_mp3(file)
        combined += sound

    combined.export(output_filename, format="mp3")
    print(f"合并完成！音频文件已保存为: {output_filename}")
    
    # 更新字幕生成状态
    conversion_status[task_id] = {
        'status': 'processing',
        'progress': 90,
        'message': "正在生成字幕文件..."
    }
    
    # 生成字幕
    if words_in_cue > 0:
        submaker.merge_cues(words_in_cue)
    
    for cue in submaker.cues:
        cue.content = cue.content.replace(' ', '')
    
    with open(srt_filename, "w", encoding="utf-8") as file:
        file.write(srt.compose(submaker.cues))
    
    # 清理临时文件
    conversion_status[task_id] = {
        'status': 'processing',
        'progress': 95,
        'message': "正在清理临时文件..."
    }
    
    print("正在清理临时文件...")
    for file in temp_files:
        try:
            os.remove(file)
        except OSError as e:
            print(f"删除文件 {file} 时出错: {e}")
    print("清理完毕")
    
    return output_filename, srt_filename


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 获取表单数据
        session['voice_id'] = request.form.get('voice', VOICES[0]['id'])
        session['rate'] = request.form.get('rate', '+0%')
        session['volume'] = request.form.get('volume', '+0%')
        session['pitch'] = request.form.get('pitch', '+0Hz')
        session['words_in_cue'] = int(request.form.get('words_in_cue', 10))
        
        # 处理文本输入或文件上传
        if 'text_input' in request.form and request.form['text_input'].strip():
            text = request.form['text_input'].strip()
        elif 'text_file' in request.files and request.files['text_file'].filename != '':
            file = request.files['text_file']
            text = file.read().decode('utf-8')  # 直接读取文件内容
        else:
            return render_template('index.html', error="请提供文本或上传文件", voices=VOICES)
        
        if not text:
            return render_template('index.html', error="文本内容为空", voices=VOICES)
        
        # 创建临时文件存储文本内容
        task_id = str(uuid.uuid4())
        text_file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{task_id}.txt")
        with open(text_file_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        # 只存储文件路径和文本长度
        session['task_id'] = task_id
        session['text_file_path'] = text_file_path
        session['text_length'] = len(text)
        
        # 初始化任务状态
        conversion_status[task_id] = {
            'status': 'queued',
            'progress': 0,
            'message': '任务已加入队列',
        }
        
        # 重定向到转换页面
        return redirect(url_for('conversion'))
    
    return render_template('index.html', voices=VOICES)


@app.route('/conversion')
def conversion():
    task_id = session.get('task_id')
    if not task_id or task_id not in conversion_status:
        return redirect(url_for('index'))
    
    # 启动转换任务
    if conversion_status[task_id]['status'] == 'queued':
        # 生成唯一文件名
        unique_id = str(uuid.uuid4())
        mp3_filename = os.path.join(app.config['OUTPUT_FOLDER'], f"output_{unique_id}.mp3")
        srt_filename = os.path.join(app.config['OUTPUT_FOLDER'], f"output_{unique_id}.srt")
        
        # 存储文件名
        conversion_status[task_id]['mp3_filename'] = mp3_filename
        conversion_status[task_id]['srt_filename'] = srt_filename
        
        # 启动异步任务
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        @copy_current_request_context
        def run_conversion():
            text_file_path = None
            try:
                # 从临时文件读取文本内容
                text_file_path = session.get('text_file_path')
                with open(text_file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                # 分割文本
                text_chunks = split_text(text, CHUNK_SIZE)
                if not text_chunks:
                    conversion_status[task_id] = {
                        'status': 'failed',
                        'progress': 100,
                        'message': '无法分割文本'
                    }
                    return
                
                # 记录开始时间
                start_time = time.time()
                
                # 运行文本转语音
                mp3_path, srt_path = loop.run_until_complete(
                    text_to_speech_chunks(
                        text_chunks, 
                        session['voice_id'], 
                        mp3_filename, 
                        srt_filename,
                        rate=session['rate'],
                        volume=session['volume'],
                        pitch=session['pitch'],
                        words_in_cue=session['words_in_cue'],
                        task_id=task_id
                    )
                )
                
                # 记录结束时间
                end_time = time.time()
                elapsed_time = end_time - start_time
                
                # 获取音频时长
                audio_duration = AudioSegment.from_mp3(mp3_path).duration_seconds
                
                # 更新任务状态为完成
                conversion_status[task_id] = {
                    'status': 'completed',
                    'progress': 100,
                    'message': '转换完成！',
                    'result': {
                        'mp3_filename': os.path.basename(mp3_path),
                        'srt_filename': os.path.basename(srt_path),
                        'elapsed_time': f"{elapsed_time:.2f}",
                        'audio_duration': f"{audio_duration:.2f}",
                        'text_length': session['text_length']
                    }
                }
            except Exception as e:
                conversion_status[task_id] = {
                    'status': 'failed',
                    'progress': 100,
                    'message': f'转换失败: {str(e)}'
                }
            finally:
                # 删除临时文本文件
                try:
                    if text_file_path and os.path.exists(text_file_path):
                        os.remove(text_file_path)
                except Exception as e:
                    print(f"删除临时文件失败: {str(e)}")
                loop.close()
        
        # 在新线程中运行转换任务
        import threading
        thread = threading.Thread(target=run_conversion)
        thread.start()
    
    return render_template('conversion.html', task_id=task_id, 
                           text_length=session['text_length'],
                           ESTIMATED_TIME_PER_1000_CHARS=ESTIMATED_TIME_PER_1000_CHARS)


@app.route('/check_status/<task_id>')
def check_status(task_id):
    if task_id in conversion_status:
        return jsonify(conversion_status[task_id])
    return jsonify({'status': 'not_found', 'message': '任务ID不存在'})


@app.route('/result/<task_id>')
def result(task_id):
    if task_id not in conversion_status:
        return redirect(url_for('index'))
    
    status = conversion_status[task_id]
    if status['status'] != 'completed':
        return redirect(url_for('conversion'))
    
    result_ = status['result']
    return render_template('result.html', 
                           mp3_filename=result_['mp3_filename'],
                           srt_filename=result_['srt_filename'],
                           elapsed_time=result_['elapsed_time'],
                           audio_duration=result_['audio_duration'],
                           text_length=result_['text_length'],
                           current_year=datetime.now().year)


@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)


@app.route('/audio/<filename>')
def serve_audio(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)


@app.context_processor
def inject_current_year():
    return {'current_year': datetime.now().year}


if __name__ == "__main__":
    # 确保在Windows上asyncio可以正常工作
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    app.run(debug=True)