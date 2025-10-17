#!/bin/bash
# VPS setup for ICT Guru bot (assumes project files already exist)
set -e

echo "=== Installing system packages ==="
apt update
apt install -y python3 python3-venv python3-pip curl build-essential

echo "=== Installing Ollama ==="
bash -c "$(curl -fsSL https://ollama.com/install.sh)"

echo "=== Pull Ollama model (mistral) ==="
ollama pull mistral

echo "=== Setup virtualenv ==="
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "=== Build vector store ==="
python build_knowledge.py

echo "=== Setup systemd service ==="
cat <<EOL | sudo tee /etc/systemd/system/ictguru.service
[Unit]
Description=ICT Guru Telegram Bot
After=network.target

[Service]
User=root
WorkingDirectory=$(pwd)
ExecStart=$(pwd)/.venv/bin/python $(pwd)/bot.py
Restart=always

[Install]
WantedBy=multi-user.target
EOL

sudo systemctl daemon-reload
sudo systemctl enable ictguru
sudo systemctl start ictguru
sudo systemctl status ictguru

echo "✅ ICT Guru bot setup complete!"
