import os
import sys
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from imc_template.bot_template import BaseBot

load_dotenv()
class TempBot(BaseBot):
    def on_orderbook(self, ob): pass
    def on_trades(self, trade): pass

bot = TempBot(os.environ["CMI_URL"], os.environ["CMI_USER"], os.environ["CMI_PASS"])
_ = bot.auth_token

print(f"PnL: {bot.get_pnl()}")
print(f"Pos: {bot.get_positions()}")

for p in ["LHR_COUNT", "TIDE_SPOT", "TIDE_SWING", "WX_SPOT", "WX_SUM", "LON_ETF", "LON_FLY", "LHR_INDEX"]:
    try:
        ob = bot.get_orderbook(p)
        if ob.buy_orders and ob.sell_orders:
            mid = (ob.buy_orders[0].price + ob.sell_orders[0].price) / 2
            print(f"{p} mid: {mid}")
    except Exception:
        pass
