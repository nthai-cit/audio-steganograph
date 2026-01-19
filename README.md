Python 3.12
`conda create -n stego python=3.12`

1. Nhúng một câu văn bản:

Bash
python main.py encode -m lsb -i "inputs/sample.wav" -o "outputs/test_text.wav" -s "Chao ban, day la mat ma!"
2. Nhúng một file ảnh (Ví dụ bạn có file image.png):

Bash
python main.py encode -m lsb -i "inputs/sample.wav" -o "outputs/test_file.wav" -s "image.png"
3. Giải mã:

Bash
python main.py decode -m lsb -i "outputs/test_text.wav"
