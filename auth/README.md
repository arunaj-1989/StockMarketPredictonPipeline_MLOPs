# TradeAssist authentication (Keycloak, local)

Local-only identity provider for the TradeAssist dashboard. **All credentials in
this folder are development defaults for a Keycloak instance that runs on your
machine — never reuse them anywhere public.**

## Start Keycloak

```powershell
docker compose up -d auth
```

Keycloak runs at <http://localhost:8080> and imports the `tradeassist` realm from
[tradeassist-realm.json](tradeassist-realm.json) on first start.

| What | Value |
|---|---|
| Admin console | <http://localhost:8080> (user `admin`, password `admin`) |
| Realm | `tradeassist` |
| OIDC client | `tradeassist-dashboard` |
| Demo user | `trader` / `Trader@123` |

## Wire the dashboard

The dashboard uses Streamlit's native OIDC login (`st.login` / `st.user`). It is
configured through `.streamlit/secrets.toml` (gitignored). Copy
`.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` — it already
matches the imported realm.

When the dashboard runs with `docker compose up dashboard`, compose mounts
[dashboard-secrets.docker.toml](dashboard-secrets.docker.toml) instead, which
reaches Keycloak through `host.docker.internal`.

If `[auth]` is missing from secrets, the dashboard runs without a login screen
(useful for tests/CI).

## Manage users

Add or manage users in the admin console: realm `tradeassist` → Users. Assign
the `trader` realm role; roles appear in the dashboard session claims.

## Resetting

Realm state persists in the `keycloak-data` volume. To re-import the realm from
scratch: `docker compose down` then `docker volume rm nifty50-mlops_keycloak-data`.
