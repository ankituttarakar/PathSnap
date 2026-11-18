# PathSnap Frontend — Laptop 2

React + Vite dashboard for real batch predictions from the FastAPI backend on
Laptop 1. It renders the backend's GPS `LineString` and each HMM/Viterbi
directed-arc feature separately. It does not contain mock predictions or
ground-truth labels.

## Requirements

- Node.js 20 or newer and npm.
- Laptop 1 and Laptop 2 on the same reachable Wi-Fi/LAN.
- Laptop 1 running the API on port 8000.

## Install and configure

From PowerShell on Laptop 2:

```powershell
Set-Location 'C:\Users\ankit\OneDrive\Desktop\PathSnap-ML\frontend'
npm install
Copy-Item .env.example .env
notepad .env
```

Set `VITE_API_BASE_URL` in `.env` to Laptop 1's actual LAN IPv4 address, for
example:

```dotenv
VITE_API_BASE_URL=http://192.168.1.20:8000
```

The address above is only an example; obtain and use Laptop 1's actual address.
Do not use `localhost` for the two-laptop demo, because that would point to
Laptop 2 itself. Restart Vite after changing `.env`.

## Start the UI

```powershell
npm run dev -- --host 0.0.0.0 --port 5173
```

Open the URL Vite prints on Laptop 2. If you also want to open the UI from
another device on the LAN, use Laptop 2's IPv4 address and allow inbound TCP
port 5173 on Laptop 2's Private firewall profile.

## Start the backend on Laptop 1

From a separate PowerShell window:

```powershell
Set-Location 'C:\Users\ankit\OneDrive\Desktop\PathSnap-ML'
.\.venv\Scripts\Activate.ps1
$env:PATHSNAP_API_HOST = '0.0.0.0'
$env:PATHSNAP_API_PORT = '8000'
$env:PATHSNAP_DEMO_PARTITION = 'train'
$env:PATHSNAP_CORS_ORIGINS = 'http://<Laptop2-IPv4>:5173'
python -m api.main
```

Replace `<Laptop2-IPv4>` with Laptop 2's actual address. CORS must allow the
exact browser origin (scheme, address, and port), not the API base URL. Multiple
origins can be comma-separated. The API intentionally exposes the configured
`train` partition for the demo, not `student_test`.

To find Laptop 1's active IPv4 address, run on Laptop 1:

```powershell
Get-NetIPConfiguration |
  Where-Object { $_.NetAdapter.Status -eq 'Up' -and $_.IPv4DefaultGateway } |
  Select-Object InterfaceAlias,
    @{Name='IPv4';Expression={$_.IPv4Address.IPAddress}},
    @{Name='Gateway';Expression={$_.IPv4DefaultGateway.NextHop}}
```

Use the IPv4 value for the Wi-Fi adapter in Laptop 2's `.env`. Verify access
from Laptop 2:

```powershell
$backendIp = '<Laptop1-IPv4>'
Test-NetConnection -ComputerName $backendIp -Port 8000
$backendUrl = "http://${backendIp}:8000"
Invoke-RestMethod "$backendUrl/health"
Invoke-RestMethod "$backendUrl/trips"
```

If `TcpTestSucceeded` is false, check that both devices are on the same
non-isolated network and allow inbound TCP port 8000 on Laptop 1's Private
Windows Firewall profile. Do not expose the API on a public network.

## API contract used by the UI

- `GET /health` — backend/model readiness and prediction mode.
- `GET /trips` — IDs in the configured demo partition.
- `POST /predict` with `{"trip_id":"00000083"}` — GPS GeoJSON, decoded arc
  GeoJSON, confidence values, and summary counts/runtime.

GeoJSON coordinates arrive as `[longitude, latitude]`; the map converts them to
Leaflet's `[latitude, longitude]`. Each predicted directed arc is rendered as
its own line, so disconnected arcs are not joined by the UI. Confidence is the
classifier probability for the selected candidate, not prediction accuracy.
No ground-truth route or accuracy is returned or displayed.

## Validation and troubleshooting

```powershell
npm test
npm run build
```

- **API URL required:** confirm `frontend/.env` exists and has Laptop 1's LAN IP;
  restart Vite after edits.
- **Backend offline / timeout:** check `Test-NetConnection`, the API process,
  the address and port, Windows Firewall, and Wi-Fi client isolation.
- **CORS error:** set Laptop 1's `PATHSNAP_CORS_ORIGINS` to the exact origin
  shown in Laptop 2's browser address bar, then restart the API.
- **No trips:** check `/health` and `/trips`; the backend must be configured for
  an available `train` partition.
- **Map tiles missing:** verify internet access to OpenStreetMap tile servers.
