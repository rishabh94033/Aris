# Aris

Autonomous Reasoning & Intelligence System. Personal assistant of Rishabh Sharma. He addresses Rishabh as Sir.

## Decisions

- Build on this Mac. Later, run the same program on the spare Linux laptop (8 GB RAM, 500 GB SSD).
- The brain is a local model, Ollama `qwen2.5:3b`, so mail and chats stay on our machines.
- Phones and laptops will connect later through Tailscale. They need internet, not the same Wi-Fi.
- CI comes after the spare laptop is running Aris.

## Where we are

Step 1: talk to Aris by typing in the terminal.

Not in this step: voice, Teams, Outlook, phone clients, deploy, CI.

## Run step 1

Ollama has to be installed and running, with the model downloaded:

```bash
brew install ollama
brew services start ollama
ollama pull qwen2.5:3b
```

Then, from this folder:

```bash
python3 -m aris
```

Type `exit` to stop.
