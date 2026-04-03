import fire
import chess
import threading
import queue
import pygame
from pathlib import Path
from play import get_previous_network

# ── Layout ───────────────────────────────────────────────────────────────────
SQ    = 80          # pixels per square
PANEL = 220         # right panel width
W     = 8 * SQ + PANEL
H     = 8 * SQ

# ── Palette ──────────────────────────────────────────────────────────────────
C_LIGHT    = (240, 217, 181)
C_DARK     = (181, 136,  99)
C_HL_L     = (205, 210, 106)   # last-move highlight, light square
C_HL_D     = (170, 162,  58)   # last-move highlight, dark square
C_SEL      = ( 20, 160,  20)   # selected-square tint
C_LEGAL    = (  0, 150,   0)   # legal-move indicator
C_BG       = ( 40,  40,  40)   # panel / window background
C_TEXT     = (220, 220, 220)
C_ACCENT   = (255, 220, 100)
C_W_PIECE  = (255, 255, 255)
C_B_PIECE  = ( 20,  20,  20)
C_SHADOW   = (  0,   0,   0)

PIECE_CHAR = {
    (chess.KING,   chess.WHITE): "♔",
    (chess.QUEEN,  chess.WHITE): "♕",
    (chess.ROOK,   chess.WHITE): "♖",
    (chess.BISHOP, chess.WHITE): "♗",
    (chess.KNIGHT, chess.WHITE): "♘",
    (chess.PAWN,   chess.WHITE): "♙",
    (chess.KING,   chess.BLACK): "♚",
    (chess.QUEEN,  chess.BLACK): "♛",
    (chess.ROOK,   chess.BLACK): "♜",
    (chess.BISHOP, chess.BLACK): "♝",
    (chess.KNIGHT, chess.BLACK): "♞",
    (chess.PAWN,   chess.BLACK): "♟",
}


class ChessGUI:
    def __init__(self, network, human_color: chess.Color = chess.WHITE):
        pygame.init()
        pygame.display.set_caption("ROLE Chess")
        self.screen = pygame.display.set_mode((W, H))
        self.clock  = pygame.time.Clock()

        # Font with chess-symbol support (try several on macOS / Linux / Win)
        self.piece_font = None
        for name in ["applesymbols", "arial unicode ms", "dejavusans",
                     "segoeuisymbol", None]:
            f = pygame.font.SysFont(name, 60, bold=False)
            if f is not None:
                self.piece_font = f
                break
        self.ui_font   = pygame.font.SysFont(None, 26)
        self.move_font = pygame.font.SysFont("courier", 19)
        self.big_font  = pygame.font.SysFont(None, 32)

        self.board        = chess.Board()
        self.network      = network
        self.human_color  = human_color
        self.selected     = None   # from-square (int) chosen by human, or None
        self.legal_dests  = []     # legal Move objects from self.selected
        self.last_move    = None   # last move played (chess.Move)
        self.move_sans    = []     # SAN strings accumulated
        self.role_thinking = False
        self._move_q      = queue.Queue()

    # ── Coordinate helpers ───────────────────────────────────────────────────

    def _sq_to_xy(self, sq: int):
        """Top-left pixel of a square (board rendered from White's POV)."""
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        return f * SQ, (7 - r) * SQ

    def _xy_to_sq(self, x: int, y: int):
        if not (0 <= x < 8 * SQ and 0 <= y < H):
            return None
        return chess.square(x // SQ, 7 - y // SQ)

    # ── Drawing ──────────────────────────────────────────────────────────────

    def _draw_board(self):
        last_from = self.last_move.from_square if self.last_move else -1
        last_to   = self.last_move.to_square   if self.last_move else -1
        legal_to  = {m.to_square for m in self.legal_dests}

        for sq in chess.SQUARES:
            f  = chess.square_file(sq)
            r  = chess.square_rank(sq)
            x, y = self._sq_to_xy(sq)
            light = (f + r) % 2 == 1

            # Base color
            if sq in (last_from, last_to):
                base = C_HL_L if light else C_HL_D
            else:
                base = C_LIGHT if light else C_DARK
            pygame.draw.rect(self.screen, base, (x, y, SQ, SQ))

            # Selected-square overlay
            if sq == self.selected:
                ov = pygame.Surface((SQ, SQ), pygame.SRCALPHA)
                ov.fill((*C_SEL, 90))
                self.screen.blit(ov, (x, y))

            # Legal-move indicators
            if sq in legal_to:
                occupied = self.board.piece_at(sq) is not None
                ov = pygame.Surface((SQ, SQ), pygame.SRCALPHA)
                if occupied:                               # capture ring
                    pygame.draw.circle(ov, (*C_LEGAL, 160),
                                       (SQ // 2, SQ // 2), SQ // 2 - 3, 6)
                else:                                      # move dot
                    pygame.draw.circle(ov, (*C_LEGAL, 150),
                                       (SQ // 2, SQ // 2), 13)
                self.screen.blit(ov, (x, y))

            # Piece
            piece = self.board.piece_at(sq)
            if piece:
                sym   = PIECE_CHAR[(piece.piece_type, piece.color)]
                fg    = C_W_PIECE if piece.color == chess.WHITE else C_B_PIECE
                # shadow for contrast
                sh = self.piece_font.render(sym, True, C_SHADOW)
                cx = x + SQ // 2 - sh.get_width() // 2
                cy = y + SQ // 2 - sh.get_height() // 2
                self.screen.blit(sh, (cx + 2, cy + 2))
                surf = self.piece_font.render(sym, True, fg)
                self.screen.blit(surf, (cx, cy))

        # Rank / file labels
        for i in range(8):
            lbl = self.ui_font.render(chess.FILE_NAMES[i], True,
                                      C_DARK if i % 2 == 0 else C_LIGHT)
            self.screen.blit(lbl, (i * SQ + SQ - lbl.get_width() - 4,
                                   H - lbl.get_height() - 4))
            lbl = self.ui_font.render(str(i + 1), True,
                                      C_LIGHT if i % 2 == 0 else C_DARK)
            self.screen.blit(lbl, (4, (7 - i) * SQ + 4))

    def _draw_panel(self):
        px = 8 * SQ
        pygame.draw.rect(self.screen, C_BG, (px, 0, PANEL, H))

        # Title
        t = self.big_font.render("ROLE  vs  Human", True, C_TEXT)
        self.screen.blit(t, (px + (PANEL - t.get_width()) // 2, 14))
        pygame.draw.line(self.screen, (80, 80, 80), (px + 10, 46), (px + PANEL - 10, 46))

        # Move list — show last ~18 half-moves
        sans   = self.move_sans
        window = 18
        start  = max(0, len(sans) - window)
        # round down to even so we start on a white move
        if start % 2:
            start -= 1
        mv_num_0 = start // 2 + 1

        y = 56
        i = start
        while i < len(sans):
            mv_num = mv_num_0 + (i - start) // 2
            white  = sans[i]
            black  = sans[i + 1] if i + 1 < len(sans) else ""
            recent = i >= len(sans) - 2
            color  = C_ACCENT if recent else C_TEXT
            line   = self.move_font.render(
                f"{mv_num:>3}. {white:<7} {black}", True, color)
            self.screen.blit(line, (px + 10, y))
            y += 22
            i += 2

        pygame.draw.line(self.screen, (80, 80, 80),
                         (px + 10, H - 54), (px + PANEL - 10, H - 54))

        # Status
        if self.role_thinking:
            status, color = "ROLE is thinking…", C_ACCENT
        elif self.board.is_checkmate():
            winner = "Black" if self.board.turn == chess.WHITE else "White"
            status, color = f"Checkmate — {winner} wins!", C_ACCENT
        elif self.board.is_stalemate():
            status, color = "Stalemate — Draw", C_TEXT
        elif self.board.is_game_over():
            status, color = f"Draw  ({self.board.result()})", C_TEXT
        elif self.board.is_check():
            turn   = "White" if self.board.turn == chess.WHITE else "Black"
            status, color = f"{turn} in check!", (255, 100, 100)
        else:
            turn   = "White" if self.board.turn == chess.WHITE else "Black"
            status, color = f"{turn} to move", C_TEXT

        s = self.ui_font.render(status, True, color)
        self.screen.blit(s, (px + (PANEL - s.get_width()) // 2, H - 38))

    # ── Game logic ───────────────────────────────────────────────────────────

    def _apply_move(self, move: chess.Move):
        san = self.board.san(move)
        self.board.push(move)
        self.move_sans.append(san)
        self.last_move  = move
        self.selected   = None
        self.legal_dests = []

    def _role_thread(self, board_copy):
        move = self.network.play(board_copy).move
        self._move_q.put(move)

    def _start_role(self):
        self.role_thinking = True
        threading.Thread(target=self._role_thread,
                         args=(self.board.copy(),),
                         daemon=True).start()

    def _is_promotion(self, from_sq: int, to_sq: int) -> bool:
        piece = self.board.piece_at(from_sq)
        if piece is None or piece.piece_type != chess.PAWN:
            return False
        return chess.square_rank(to_sq) in (0, 7)

    def _handle_click(self, x: int, y: int):
        sq = self._xy_to_sq(x, y)
        if sq is None:
            return

        if self.selected is None:
            piece = self.board.piece_at(sq)
            if piece and piece.color == self.human_color:
                self.selected    = sq
                self.legal_dests = [m for m in self.board.legal_moves
                                    if m.from_square == sq]
        else:
            promo = chess.QUEEN if self._is_promotion(self.selected, sq) else None
            move  = chess.Move(self.selected, sq, promotion=promo)
            if move in self.board.legal_moves:
                self._apply_move(move)
                if not self.board.is_game_over():
                    self._start_role()
            else:
                # re-select own piece, or deselect
                piece = self.board.piece_at(sq)
                if piece and piece.color == self.human_color:
                    self.selected    = sq
                    self.legal_dests = [m for m in self.board.legal_moves
                                        if m.from_square == sq]
                else:
                    self.selected    = None
                    self.legal_dests = []

    # ── Main loop ────────────────────────────────────────────────────────────

    def run(self):
        if self.human_color == chess.BLACK:
            self._start_role()

        while True:
            self.clock.tick(30)

            # Collect ROLE's move if ready
            try:
                role_move = self._move_q.get_nowait()
                self.role_thinking = False
                self._apply_move(role_move)
            except queue.Empty:
                pass

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    pygame.quit()
                    return
                if (event.type == pygame.MOUSEBUTTONDOWN
                        and event.button == 1
                        and not self.board.is_game_over()
                        and self.board.turn == self.human_color
                        and not self.role_thinking):
                    self._handle_click(*event.pos)

            self.screen.fill(C_BG)
            self._draw_board()
            self._draw_panel()
            pygame.display.flip()


# ── Entry point ──────────────────────────────────────────────────────────────

def main(net: str,
         checkpoint_dir: Path = Path("./checkpoints"),
         expert_path: Path = Path("./stockfish"),
         depth: int = 2,
         human_color: str = "white"):

    network = get_previous_network(net, 384, checkpoint_dir, None)
    network.depth = depth

    color = chess.WHITE if human_color.lower() == "white" else chess.BLACK
    ChessGUI(network, human_color=color).run()


if __name__ == "__main__":
    fire.Fire(main)
