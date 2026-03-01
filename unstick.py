import os
import sys
import time
from dotenv import load_dotenv

def liquidate_all():
    load_dotenv()
    url = os.environ.get("CMI_URL", "http://ec2-52-19-74-159.eu-west-1.compute.amazonaws.com/")
    user = os.environ.get("CMI_USER")
    pwd  = os.environ.get("CMI_PASS")
    
    if not user or not pwd:
        print("Error: Missing CMI_USER or CMI_PASS in environment.", file=sys.stderr)
        sys.exit(1)

    # Use the official bot template to handle authentication and requests securely
    from imc_template.bot_template import BaseBot, OrderRequest, Side
    
    class TempBot(BaseBot):
        def on_orderbook(self, ob): pass
        def on_trades(self, trade): pass
        
    print("Authenticating...")
    try:
        bot = TempBot(url, user, pwd)
        _ = bot.auth_token  # trigger auth and cache bearer token
    except Exception as e:
        print(f"Auth failed: {e}")
        return
        
    print("Cancelling all active orders...")
    bot.cancel_all_orders()
    time.sleep(1)

    print("\nFetching portfolio positions...")
    try:
        positions = bot.get_positions()
    except Exception as e:
        print(f"Failed to fetch positions: {e}")
        return
        
    for product, qty in positions.items():
        if qty == 0:
            continue
            
        side = Side.SELL if qty > 0 else Side.BUY
        vol  = abs(qty)
        
        try:
            ob = bot.get_orderbook(product)
        except Exception as e:
            print(f"[{product}] Failed to get orderbook: {e}")
            continue
            
        px = 0
        if side == Side.SELL and ob.buy_orders:
            px = int(ob.buy_orders[0].price) - 1 # Hit the bid heavily
        elif side == Side.BUY and ob.sell_orders:
            px = int(ob.sell_orders[0].price) + 1 # Hit the ask heavily
            
        if px <= 0:
            print(f"[{product}] Skipping {qty} position (No liquidity to cross!)")
            continue
            
        print(f"[{product}] Dumping {vol} units via {side.name} @ {px} ...")
        res = bot.send_order(OrderRequest(product=product, price=px, side=side, volume=vol))
        if not res:
            print(f"[{product}] Failed to send order.")
        time.sleep(0.2)
        
    print("\n✅ Liquidation complete. Your positions should be near zero.")

if __name__ == "__main__":
    liquidate_all()
