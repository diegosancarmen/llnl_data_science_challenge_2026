# Stage 4 Interactive Dashboard

This workflow runs the FastAPI broker, browser-based PyVista viewer, and
Streamlit dashboard from a native Windows Conda environment. The CT volume is
read by the PyVista viewer; FastAPI serves metadata, chat, and shared strut
selection state.

## 1. Create the Windows environment

Run these commands in Windows PowerShell. The repository is stored in WSL,
so use its Windows network path:

```powershell
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process

cd \\wsl.localhost\Ubuntu\home\hannahdc\llnl_data_science_challenge_2026

conda create --prefix .\dssi_env_win python=3.11 -y
.\dssi_env_win\Scripts\activate

pip install -r part2\stage_4_interactive_dashboard_napari_chatbot\windows_requirements.txt
```

If PowerShell reports that the environment is already present, skip the
`conda create` command and activate it with:

```powershell
.\dssi_env_win\Scripts\activate
```

## 2. Start the FastAPI broker

Open a new PowerShell window, repeat the environment activation and repository
directory commands above, then run:

```powershell
uvicorn part2.stage_4_interactive_dashboard_napari_chatbot.run_fastapi:app --host 127.0.0.1 --port 8000
```

The broker provides the REST and WebSocket endpoints used by the viewer and
dashboard. Leave this window running.

Final dashboard review decisions are persisted by the broker in
`part2/stage_4_interactive_dashboard_napari_chatbot/output/review_decisions.json`.
Set `DASHBOARD_REVIEW_DECISIONS_JSON` before starting FastAPI to use a different
shared location.

To enable Gemini-powered open-ended questions, set both `GEMINI_ENABLED=true`
and `GEMINI_API_KEY` before starting FastAPI. `GEMINI_MODEL` optionally
overrides the default `gemini-2.5-flash` model. Basic CSV count/filter/select
questions stay local and do not consume Gemini quota; the dashboard toggle is
off by default. The chat assistant cannot save final review choices.

## 3. Start the browser PyVista viewer

Open another PowerShell window, activate the environment, change to the
repository directory, and run:

```powershell
python -m part2.napari_visualizer.web_visualizer --api-url http://127.0.0.1:8000
```

The viewer normally serves the embedded two-pane visualization at
`http://127.0.0.1:8081`. It may take time to build the initial CT scene; wait
for `initialization complete` in the viewer window.

For a Windows/WSL setup where Streamlit cannot reach the default viewer bind
address, use:

```powershell
python -m part2.napari_visualizer.web_visualizer --host 0.0.0.0 --api-url http://127.0.0.1:8000
```

## 4. Start the Streamlit dashboard

Open a third PowerShell window, activate the environment, change to the
repository directory, and run:

```powershell
streamlit run part2/stage_4_interactive_dashboard_napari_chatbot/dashboard/app_reactive_v2.py
```

Open the URL printed by Streamlit, normally
`http://127.0.0.1:8501`. The dashboard includes defect filters, station
trends, raw data, chat, and the embedded PyVista viewer.

The dashboard uses FastAPI at `http://127.0.0.1:8000` by default. To use a
different broker URL, set `DASHBOARD_API_URL` before starting Streamlit:

```powershell
$env:DASHBOARD_API_URL = "http://127.0.0.1:8000"
streamlit run part2/stage_4_interactive_dashboard_napari_chatbot/dashboard/app_reactive_v2.py
```

## Optional: terminal chatbot

With FastAPI running, start the terminal chatbot from an activated PowerShell
window in the repository root:

```powershell
python .\part2\stage_4_interactive_dashboard_napari_chatbot\run_agent_cli.py
```

The chatbot talks directly to FastAPI and can answer defect-summary questions,
inspect struts, and broadcast strut selections to the connected viewer and
dashboard.

Get-Process python | Stop-Process -Force      