package metalmc.bench;

import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;

public final class BenchClient implements ClientModInitializer {
    @Override
    public void onInitializeClient() {
        if (!Bench.ENABLED) return;
        ClientTickEvents.END_CLIENT_TICK.register(Bench::onTick);
    }
}
