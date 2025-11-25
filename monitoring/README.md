# Modul Monitoring Mining Operations

Sistem ini memantau status alat, produksi, dan bahan bakar secara real-time.

## Cara Instalasi

1. Install library: `pip install -r requirements.txt`
2. Jalankan Exporter: `python mining_exporter.py`
3. Jalankan Prometheus: `prometheus.exe --config.file=prometheus.yml`
4. Jalankan Grafana & Login (admin/admin).

## Cara Import Dashboard

1. Di Grafana, ke menu **Dashboards** > **New** > **Import**.
2. Upload file `dashboard.json` dari folder ini.

## Cara Setting Notifikasi (Alerting)

Karena alasan keamanan token, Alert harus disetting manual:

1. Buat Bot Telegram via @BotFather.
2. Di Grafana: Menu **Alerting** > **Contact Points**.
3. Tambahkan **Telegram** > Masukkan Token Bot & Chat ID kamu.
4. Klik **Test** untuk memastikan notifikasi jalan.
