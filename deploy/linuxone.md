# Run it on IBM LinuxONE

Fill these two values in once, then copy the commands.

```bash
KEY=~/path/to/your-linuxone-key.pem      # private key, never share it
HOST=linux1@YOUR.VM.IP                   # user@ip from the LinuxONE Community Cloud dashboard
```

1. Copy the code. It's small, and data and runs are excluded.

```bash
tar --exclude=data --exclude=runs --exclude=.git -czf /tmp/eeg.tgz -C .. eeg-review-queue
scp -i "$KEY" /tmp/eeg.tgz "$HOST":~
ssh -i "$KEY" "$HOST" 'tar xzf eeg.tgz && cd eeg-review-queue && bash deploy/setup.sh'
```

2. Download data on the mainframe, which is faster than uploading it, then evaluate.

```bash
ssh -i "$KEY" "$HOST" 'cd eeg-review-queue && .venv/bin/python scripts/download_chbmit.py && .venv/bin/python -m src.detector eval | tee runs/eval_linuxone.txt'
```

3. Start the live app on LinuxONE.

```bash
ssh -i "$KEY" "$HOST" 'cd eeg-review-queue && nohup .venv/bin/python app/server.py --real > runs/server.log 2>&1 &'
```

4. View it in your laptop browser through an SSH tunnel. The app still runs on the mainframe.

```bash
ssh -i "$KEY" -N -L 8050:localhost:8050 "$HOST"
```

Then open http://localhost:8050

5. Proof shots for the video, run in an SSH session:

```bash
uname -m
lscpu | head -15
ps aux | grep server.py
tail -5 runs/eval_linuxone.txt
```
