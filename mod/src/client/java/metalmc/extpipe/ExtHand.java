package metalmc.extpipe;

import com.mojang.blaze3d.vertex.PoseStack;
import metalmc.backend.MetalExtPipe;
import net.minecraft.client.Minecraft;
import net.minecraft.client.renderer.GameRenderer;
import net.minecraft.client.renderer.SubmitNodeStorage;
import net.minecraft.client.renderer.state.GameRenderState;
import net.minecraft.client.renderer.state.level.CameraRenderState;
import net.minecraft.client.renderer.state.level.PlayerRenderState;
import net.minecraft.world.level.GameType;
import org.joml.Matrix4f;

/**
 * The first-person hand in the external pipeline (metalmc.backend.MetalExtPipe): an OptiFine-style pipeline wants it in
 * its G-buffer before its deferred passes, but vanilla draws it in a pass of its own after the level. When the
 * description asks for it (hand routes), the hands and held items are submitted with the level's own features instead
 * (LevelRendererExtMixin, before the level prepares them: the level's feature frame is in use for the whole main pass,
 * so the hand can't have one of its own there), drawn with the level's projection and bobbing, and vanilla's hand pass
 * is skipped that frame (GameRendererExtMixin).
 */
public final class ExtHand {
    private ExtHand() {
    }

    /** The hand went into the level's features this frame: vanilla's own hand pass is skipped. */
    public static boolean drawn;

    public static void submit(GameRenderer gameRenderer, SubmitNodeStorage storage) {
        drawn = false;
        GameRenderState state = gameRenderer.gameRenderState();
        CameraRenderState cam = state.levelRenderState.cameraRenderState;
        PlayerRenderState player = state.levelRenderState.playerRenderState;
        Minecraft mc = Minecraft.getInstance();
        if (cam.isPanoramicMode || !player.hasPlayer || !state.optionsRenderState.cameraType.isFirstPerson()
            || cam.entityRenderState.isSleeping || state.guiRenderState.isHudHidden
            || mc.gameMode == null || mc.gameMode.getPlayerMode() == GameType.SPECTATOR || !MetalExtPipe.handWanted()) {
            return;
        }
        // renderItemInHand's pose: the view rotation undone (the level's model-view applies it again).
        PoseStack poseStack = new PoseStack();
        poseStack.mulPose(cam.viewRotationMatrix.invert(new Matrix4f()));
        gameRenderer.firstPersonHandsAndItemsRenderer.submitHandsWithItems(cam.cameraEntityPartialTicks, poseStack, storage, player,
            player.firstPersonHandsAndItems);
        drawn = true;
    }
}
