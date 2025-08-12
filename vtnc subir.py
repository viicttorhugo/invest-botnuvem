# app_bot.py
import os, json, time, threading, asyncio, requests
from datetime import datetime, time as dt_time
from flask import Flask, request, session, redirect, url_for, render_template_string, jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pandas_ta as ta
import websockets
from iqoptionapi.stable_api import IQ_Option

# =================== CONFIG BÁSICA ===================
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "Vhj02122024$")

API_AUTH_BASE = os.getenv("API_AUTH_BASE", "https://licencas-2.onrender.com")
API_AUTH_URL  = f"{API_AUTH_BASE}/api/auth"

PAYOUT = 0.9
MAX_MG = 6
ATIVOS = {"EURUSD": "frxEURUSD", "EURJPY": "frxEURJPY", "USDJPY": "frxUSDJPY"}
HORARIOS = [(dt_time(8,0), dt_time(10,0)),
            (dt_time(10,30), dt_time(12,0)),
            (dt_time(12,30), dt_time(14,30)),
            (dt_time(15,0), dt_time(16,30))]

# Estado por usuário logado
USERS_STATE = {}  # user -> dict(state)

def _init_user_state(user):
    if user not in USERS_STATE:
        USERS_STATE[user] = {
            "executando": False,
            "stakes_iniciais": {},
            "stakes_atuais": {},
            "martingale_niveis": {},
            "stop_gain": {},
            "stop_loss": {},
            "saldo_acumulado": {},
            "modelos": {},
            "iq": None,
            "deriv_token": None,
            "telegram_bot": None,
            "telegram_chat": None,
            "logs": [],
            "thread": None
        }

def log(user, msg, tag="info"):
    ts = datetime.now().strftime("[%H:%M:%S]")
    line = f"{ts} {msg}"
    USERS_STATE[user]["logs"].append({"t": tag, "m": line})
    # limita
    if len(USERS_STATE[user]["logs"]) > 1000:
        USERS_STATE[user]["logs"] = USERS_STATE[user]["logs"][-1000:]

def horario_permitido():
    agora = datetime.now().time()
    return any(a <= agora <= b for a,b in HORARIOS)

async def deriv_buy(token, symbol, amount, contract_type, duration):
    try:
        async with websockets.connect('wss://ws.derivws.com/websockets/v3?app_id=1089') as ws:
            await ws.send(json.dumps({"authorize": token}))
            await ws.recv()
            await ws.send(json.dumps({"proposal": 1, "amount": amount, "basis": "stake", "contract_type": contract_type,
                                       "currency": "USD", "duration": duration, "duration_unit": "m", "symbol": symbol}))
            proposal = json.loads(await ws.recv())
            if 'error' in proposal:
                return None, proposal['error']['message']
            pid = proposal['proposal']['id']
            await ws.send(json.dumps({"buy": pid, "price": amount}))
            result = json.loads(await ws.recv())
            if 'error' in result:
                return None, result['error']['message']
            return result.get('buy', {}).get('contract_id'), None
    except Exception as e:
        return None, str(e)

async def check_result(token, cid):
    try:
        async with websockets.connect('wss://ws.derivws.com/websockets/v3?app_id=1089') as ws:
            await ws.send(json.dumps({"authorize": token}))
            await ws.recv()
            await ws.send(json.dumps({"proposal_open_contract": 1, "contract_id": cid}))
            data = json.loads(await ws.recv())
            return data.get('proposal_open_contract', {}).get('status')
    except Exception:
        return None

def calcular_indicadores(df):
    df['rsi'] = ta.rsi(df['close'], length=14)
    macd = ta.macd(df['close'])
    if isinstance(macd, pd.DataFrame):
        df['macd'] = macd.get('MACD_12_26_9', macd.iloc[:,0])
        df['macd_signal'] = macd.get('MACDs_12_26_9', macd.iloc[:,1] if macd.shape[1] > 1 else macd.iloc[:,0])
        df['macd_hist'] = macd.get('MACDh_12_26_9', macd.iloc[:,2] if macd.shape[1] > 2 else 0)
    bb = ta.bbands(df['close'])
    if isinstance(bb, pd.DataFrame) and bb.shape[1] >= 3:
        df['bb_lower'], df['bb_middle'], df['bb_upper'] = bb.iloc[:,0], bb.iloc[:,1], bb.iloc[:,2]
    df['ma10'] = ta.sma(df['close'], length=10)
    df['ma20'] = ta.sma(df['close'], length=20)
    df['ma50'] = ta.sma(df['close'], length=50)
    df['ma_cross_10_20'] = (df['ma10'] > df['ma20']).astype(int)
    df['ma_cross_20_50'] = (df['ma20'] > df['ma50']).astype(int)
    stoch = ta.stoch(df['high'], df['low'], df['close'])
    if isinstance(stoch, pd.DataFrame) and stoch.shape[1] >= 2:
        df['stoch_k'], df['stoch_d'] = stoch.iloc[:,0], stoch.iloc[:,1]
    adx = ta.adx(df['high'], df['low'], df['close'])
    if isinstance(adx, pd.DataFrame) and 'ADX_14' in adx.columns:
        df['adx'] = adx['ADX_14']
    df['willr'] = ta.willr(df['high'], df['low'], df['close'])
    return df.dropna()

def enviar_telegram(bot_token, chat_id, text):
    if not bot_token or not chat_id:
        return
    try:
        requests.post(f"https://api.telegram.org/bot{bot_token}/sendMessage",
                      data={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"})
    except:
        pass

async def analisar_e_executar(user, nome, symbol, modelo, scaler):
    st = USERS_STATE[user]
    if st["saldo_acumulado"].get(nome, 0) >= st["stop_gain"].get(nome, float("inf")) or \
       st["saldo_acumulado"].get(nome, 0) <= st["stop_loss"].get(nome, -float("inf")):
        log(user, f"{nome} - Stop atingido. Não operando.", "aviso"); return
    if not horario_permitido():
        log(user, f"{nome} - Fora do horário permitido.", "aviso"); return

    iq = st["iq"]
    candles = iq.get_candles(nome, 900, 500, time.time())
    df = pd.DataFrame(candles).rename(columns={'max':'high','min':'low'})
    df = calcular_indicadores(df)
    if df.empty:
        log(user, f"{nome} - Indicadores não calculados.", "aviso"); return

    ult = df.iloc[-1:]
    features = ['rsi','macd','macd_signal','macd_hist','bb_lower','bb_middle','bb_upper',
                'ma10','ma20','ma50','ma_cross_10_20','ma_cross_20_50','stoch_k','stoch_d','adx','willr']
    if any(f not in ult.columns for f in features):
        log(user, f"{nome} - Feature faltando.", "aviso"); return

    X = scaler.transform(ult[features])
    prob = modelo.predict_proba(X)[0][1] * 100
    direcao = 'CALL' if prob > 50 else 'PUT'
    confianca = prob if prob > 50 else 100 - prob
    log(user, f"{nome} - Confiança: {confianca:.1f}% Direção: {direcao}", "info")

    if confianca >= 80:
        stake = st["stakes_atuais"].get(nome, 1.0)
        log(user, f"{nome} - Enviando sinal... Stake ${stake:.2f}", "info")

        cid, err = await deriv_buy(st["deriv_token"], symbol, stake, direcao, 15)
        if err:
            log(user, f"{nome} - Erro: {err}", "aviso")
            enviar_telegram(st["telegram_bot"], st["telegram_chat"], f"Erro {nome}: {err}")
            return

        enviar_telegram(
            st["telegram_bot"], st["telegram_chat"],
            f"📢 *Sinal Detectado: {nome}*\n- *Direção:* `{direcao}`\n- *Stake:* `${stake:.2f}`\n- *Confiança:* `{confianca:.1f}%`\n- *Horário:* `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`"
        )

        await asyncio.sleep(900)
        status = await check_result(st["deriv_token"], cid)
        if status == "won":
            ganho = stake * PAYOUT
            st["saldo_acumulado"][nome] = st["saldo_acumulado"].get(nome, 0) + ganho
            st["stakes_atuais"][nome] = st["stakes_iniciais"].get(nome, stake)
            st["martingale_niveis"][nome] = 0
            log(user, f"{nome} - GAIN +${ganho:.2f}", "gain")
            enviar_telegram(st["telegram_bot"], st["telegram_chat"], f"✅ *Resultado: GAIN ({nome})* +${ganho:.2f}")
        else:
            st["saldo_acumulado"][nome] = st["saldo_acumulado"].get(nome, 0) - stake
            if st["martingale_niveis"].get(nome, 0) < MAX_MG:
                st["stakes_atuais"][nome] = st["stakes_atuais"].get(nome, stake) * 2
                st["martingale_niveis"][nome] = st["martingale_niveis"].get(nome, 0) + 1
            else:
                st["stakes_atuais"][nome] = st["stakes_iniciais"].get(nome, stake)
                st["martingale_niveis"][nome] = 0
            log(user, f"{nome} - LOSS -${stake:.2f}", "loss")
            enviar_telegram(st["telegram_bot"], st["telegram_chat"], f"❌ *Resultado: LOSS ({nome})* -${stake:.2f}")

def _treinar_modelos(iq):
    modelos = {}
    for nome in ATIVOS:
        try:
            candles = iq.get_candles(nome, 900, 500, time.time())
            df = pd.DataFrame(candles).rename(columns={'max':'high','min':'low'})
            df = calcular_indicadores(df)
            if df.empty: continue
            df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
            X = df[['rsi','macd','macd_signal','macd_hist','bb_lower','bb_middle','bb_upper',
                    'ma10','ma20','ma50','ma_cross_10_20','ma_cross_20_50','stoch_k','stoch_d','adx','willr']].dropna()
            y = df['target'].loc[X.index]
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)
            modelo = RandomForestClassifier(n_estimators=100, random_state=42)
            modelo.fit(Xs, y)
            modelos[nome] = (modelo, scaler)
        except Exception:
            pass
    return modelos

def worker(user):
    st = USERS_STATE[user]
    try:
        log(user, "Conectando IQ Option...", "info")
        iq = IQ_Option(st["IQ_EMAIL"], st["IQ_PASSWORD"])
        iq.connect()
        iq.change_balance(st["IQ_ACCOUNT"])
        st["iq"] = iq
        st["modelos"] = _treinar_modelos(iq)
        log(user, "Modelos treinados.", "info")
        enviar_telegram(st["telegram_bot"], st["telegram_chat"], "✅ Bot web ativado!")

        while st["executando"]:
            now = datetime.now()
            if now.minute % 15 == 14 and now.second >= 25 and horario_permitido():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                tasks = []
                for nome, symbol in ATIVOS.items():
                    if nome in st["modelos"]:
                        modelo, scaler = st["modelos"][nome]
                        tasks.append(analisar_e_executar(user, nome, symbol, modelo, scaler))
                if tasks:
                    loop.run_until_complete(asyncio.gather(*tasks))
                time.sleep(60)
            else:
                time.sleep(1)
    except Exception as e:
        log(user, f"Erro no worker: {e}", "aviso")
    finally:
        st["executando"] = False
        log(user, "Worker finalizado.", "info")

# =================== ROTAS ===================
TPL_LOGIN = """
<!doctype html><html><head>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
<title>Login Bot</title></head><body class="p-4">
<div class="container" style="max-width:420px">
  <h3 class="mb-3">Login</h3>
  <form method="post">
    <div class="mb-3"><label class="form-label">Usuário</label>
      <input name="usuario" class="form-control" required></div>
    <div class="mb-3"><label class="form-label">Senha</label>
      <input name="senha" type="password" class="form-control" required></div>
    <button class="btn btn-primary w-100">Entrar</button>
  </form>
  {% if erro %}<div class="alert alert-danger mt-3">{{erro}}</div>{% endif %}
</div></body></html>
"""

TPL_PAINEL = """
<!doctype html><html><head>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
<title>Painel do Bot</title></head><body class="p-3">
<div class="container">
  <div class="d-flex justify-content-between align-items-center mb-2">
    <h4>Bem-vindo, {{user}}</h4>
    <a class="btn btn-outline-secondary btn-sm" href="{{url_for('logout')}}">Sair</a>
  </div>
  <form class="row g-2" method="post" action="{{url_for('start_bot')}}">
    <div class="col-12"><h6>Configurações</h6></div>
    <div class="col-md-3"><input class="form-control" name="iq_email" placeholder="IQ Email" required></div>
    <div class="col-md-3"><input class="form-control" name="iq_password" type="password" placeholder="IQ Senha" required></div>
    <div class="col-md-2">
      <select name="iq_account" class="form-select">
        <option value="PRACTICE">PRACTICE</option>
        <option value="REAL">REAL</option>
      </select>
    </div>
    <div class="col-md-4"><input class="form-control" name="deriv_token" placeholder="Deriv Token" required></div>
    <div class="col-md-4"><input class="form-control" name="tg_bot" placeholder="Telegram Bot API (opcional)"></div>
    <div class="col-md-4"><input class="form-control" name="tg_chat" placeholder="Telegram Chat ID (opcional)"></div>

    <div class="col-12 mt-3"><h6>Parâmetros por par</h6></div>
    {% for par in pares %}
      <div class="col-md-2"><label class="form-label">{{par}}</label></div>
      <div class="col-md-3"><input class="form-control" name="stake_{{par}}" placeholder="Stake" required></div>
      <div class="col-md-3"><input class="form-control" name="gain_{{par}}" placeholder="Stop Gain" required></div>
      <div class="col-md-3"><input class="form-control" name="loss_{{par}}" placeholder="Stop Loss" required></div>
    {% endfor %}
    <div class="col-12">
      <button class="btn btn-success">Iniciar Bot</button>
      <a class="btn btn-danger" href="{{url_for('stop_bot')}}">Encerrar Bot</a>
    </div>
  </form>

  <hr>
  <h6>Logs</h6>
  <pre id="logs" style="background:#111;color:#eee;padding:12px;border-radius:8px;height:340px;overflow:auto"></pre>
</div>
<script>
async function fetchLogs(){
  const r = await fetch("{{url_for('logs')}}");
  const data = await r.json();
  const el = document.getElementById("logs");
  el.textContent = data.logs.map(x=>x.m).join("\\n");
  el.scrollTop = el.scrollHeight;
}
setInterval(fetchLogs, 1500);
fetchLogs();
</script>
</body></html>
"""

@app.route("/", methods=["GET","POST"])
def login():
    if request.method == "POST":
        usuario = request.form.get("usuario","").strip()
        senha   = request.form.get("senha","").strip()
        try:
            r = requests.post(API_AUTH_URL, json={"usuario": usuario, "senha": senha}, timeout=8)
            if r.status_code == 200 and r.json().get("status") == "ok":
                session["user"] = usuario
                _init_user_state(usuario)
                return redirect(url_for("painel"))
            else:
                return render_template_string(TPL_LOGIN, erro="Acesso negado")
        except Exception as e:
            return render_template_string(TPL_LOGIN, erro=f"Erro na autenticação: {e}")
    return render_template_string(TPL_LOGIN, erro=None)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/painel")
def painel():
    if "user" not in session: return redirect(url_for("login"))
    return render_template_string(TPL_PAINEL, user=session["user"], pares=list(ATIVOS.keys()))

@app.route("/start", methods=["POST"])
def start_bot():
    if "user" not in session: return redirect(url_for("login"))
    user = session["user"]; _init_user_state(user)
    st = USERS_STATE[user]
    if st["executando"]:
        return redirect(url_for("painel"))

    # Credenciais e parâmetros
    st["IQ_EMAIL"] = request.form["iq_email"].strip()
    st["IQ_PASSWORD"] = request.form["iq_password"].strip()
    st["IQ_ACCOUNT"] = request.form.get("iq_account","PRACTICE")
    st["deriv_token"] = request.form["deriv_token"].strip()
    st["telegram_bot"] = request.form.get("tg_bot","").strip()
    st["telegram_chat"] = request.form.get("tg_chat","").strip()

    try:
        st["stakes_iniciais"].clear(); st["stakes_atuais"].clear()
        st["martingale_niveis"].clear(); st["stop_gain"].clear(); st["stop_loss"].clear()
        st["saldo_acumulado"].clear()

        for par in ATIVOS.keys():
            stake = max(float(request.form[f"stake_{par}"]), 0.5)
            sg = float(request.form[f"gain_{par}"])
            sl = float(request.form[f"loss_{par}"])
            st["stakes_iniciais"][par] = stake
            st["stakes_atuais"][par] = stake
            st["stop_gain"][par] = sg
            st["stop_loss"][par] = sl
            st["saldo_acumulado"][par] = 0
            st["martingale_niveis"][par] = 0
    except Exception as e:
        log(user, f"Valores inválidos: {e}", "aviso")
        return redirect(url_for("painel"))

    st["executando"] = True
    th = threading.Thread(target=worker, args=(user,), daemon=True)
    st["thread"] = th
    th.start()
    log(user, "Bot iniciado.", "info")
    return redirect(url_for("painel"))

@app.route("/stop")
def stop_bot():
    if "user" not in session: return redirect(url_for("login"))
    st = USERS_STATE[session["user"]]
    st["executando"] = False
    log(session["user"], "Solicitada parada do bot.", "aviso")
    return redirect(url_for("painel"))

@app.route("/logs")
def logs():
    if "user" not in session: return jsonify({"logs":[]})
    return jsonify({"logs": USERS_STATE[session["user"]]["logs"]})

# Health
@app.route("/health")
def health():
    return "ok"

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)

