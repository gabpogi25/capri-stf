from manim import *
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Custom Premium Color Palette
THEME_CYAN = "#00E5FF"
THEME_GREEN = "#00E676"
THEME_RED = "#FF1744"
THEME_YELLOW = "#FFC400"
THEME_BG = "#0F0F0F"

config.background_color = THEME_BG

# ---------------------------------------------------------------------------
# SCENE 1: The Problem & Degeneracies (Slides 1-11)
# ---------------------------------------------------------------------------
class Scene1_Problem(Scene):
    def construct(self):
        # Slide 1: Blank
        self.wait(0.5)

        # Slide 2: Scattered photos using real images
        base_dir = str(PROJECT_ROOT / r"data/real_images/image-matching-challenge-2025/train/pt_sacrecoeur_trevi_tajmahal")
        img_path1 = f"{base_dir}\\taj_mahal_02760416_4808737899.png"
        img_path2 = f"{base_dir}\\taj_mahal_03779515_11822031593.png"
        img_path3 = f"{base_dir}\\taj_mahal_05099279_2123782505.png"
        img_path4 = f"{base_dir}\\taj_mahal_05644193_4809375226.png"
        img_path5 = f"{base_dir}\\taj_mahal_07292828_6865863886.png"

        cam1 = ImageMobject(img_path1).set_height(2).shift(LEFT*4 + DOWN*2)
        cam2 = ImageMobject(img_path2).set_height(2).shift(RIGHT*4 + DOWN*2)
        cam3 = ImageMobject(img_path3).set_height(2).shift(UP*2 + LEFT*1)
        photos = Group(cam1, cam2, cam3)
        self.play(FadeIn(photos, shift=UP))

        # Slide 3: Lines between them (Feature matches)
        # Approximate coordinates for the main dome (top center) and its water reflection (bottom center)
        dome1 = cam1.get_center() + UP * 0.4
        dome2 = cam2.get_center() + UP * 0.4
        dome3 = cam3.get_center() + UP * 0.4
        
        reflection1 = cam1.get_center() + DOWN * 0.3

        # Green dashed lines representing correct physical matches (dome to dome)
        good_lines = VGroup(
            DashedLine(dome1, dome2, color=THEME_GREEN, stroke_width=4),
            DashedLine(dome2, dome3, color=THEME_GREEN, stroke_width=4),
            DashedLine(dome3, dome1, color=THEME_GREEN, stroke_width=4)
        )
        self.play(Create(good_lines))

        # Slide 4 & 5: The Degeneracy (Red line matching real dome to water reflection)
        bad_line = DashedLine(dome2, reflection1, color=THEME_RED, stroke_width=6)
        
        self.play(Create(bad_line))
        
        # Slide 6: Zoom in to highlight the false match
        zoom_group = Group(cam1, cam2, good_lines, bad_line)
        self.play(zoom_group.animate.scale(1.5).shift(UP*1 + LEFT*1))

        # Slide 7: Put zoom component aside, highlight section
        self.play(zoom_group.animate.scale(0.5).to_edge(LEFT))
        extract_box = SurroundingRectangle(cam1, color=THEME_YELLOW, buff=0.1)
        self.play(Create(extract_box))

        # Slide 8: Extract "reflection in a puddle"
        img_puddle = ImageMobject(img_path1).set_width(3).to_edge(RIGHT)
        text_puddle = Text("Reflection\nin Puddle", font_size=24, color=THEME_RED).next_to(img_puddle, DOWN)
        self.play(FadeIn(img_puddle), FadeIn(text_puddle))

        # Slide 9: "pane of glass"
        img_glass = ImageMobject(img_path2).set_width(3).to_edge(RIGHT)
        text_glass = Text("Pane of Glass", font_size=24, color=THEME_RED).next_to(img_glass, DOWN)
        self.play(FadeOut(img_puddle), FadeIn(img_glass), Transform(text_puddle, text_glass))

        # Slide 10: "repeating texture"
        img_texture = ImageMobject(img_path3).set_width(3).to_edge(RIGHT)
        text_texture = Text("Repeating\nTexture", font_size=24, color=THEME_RED).next_to(img_texture, DOWN)
        self.play(FadeOut(img_glass), FadeIn(img_texture), Transform(text_puddle, text_texture))

        # Slide 11: Fill with random degenerate points (using actual images)
        images_list = [img_path1, img_path2, img_path3, img_path4, img_path5]
        degens = Group(*[
            ImageMobject(images_list[i % 5]).set_width(1).move_to(
                img_texture.get_center() + np.random.uniform(-1.5, 1.5, 3)
            ) for i in range(8)
        ])
        self.play(FadeOut(text_puddle), FadeOut(img_texture), FadeIn(degens))
        self.wait(1)


# ---------------------------------------------------------------------------
# SCENE 2: Title and Formula (Slides 12-16)
# ---------------------------------------------------------------------------
class Scene2_Title(Scene):
    def construct(self):
        # Slide 12: Title of study
        title_full = Text(
            "Conditioning-Aware Projective Invariants for\nCalibration-Free 3D Reconstruction Under\nSparse and Non-Rigid Conditions via\n6-Point Syzygy Residuals in P3 Feature Certification",
            font_size=24
        )
        self.play(FadeIn(title_full))
        self.wait(1)

        # Slide 13: Pull out C A P R I
        c = Text("C", color=THEME_CYAN, font_size=60)
        a = Text("A", color=THEME_CYAN, font_size=60)
        p = Text("P", color=THEME_CYAN, font_size=60)
        r = Text("R", color=THEME_CYAN, font_size=60)
        i = Text("I", color=THEME_CYAN, font_size=60)
        
        capri_word = VGroup(c, a, p, r, i).arrange(RIGHT, buff=0.1)
        self.play(FadeOut(title_full), FadeIn(capri_word))

        # Slide 14 & 15: Convert to formula and highlight components
        formula = VGroup(
            Text("p", font_size=48),
            Text("=", font_size=48),
            Text("SUM", font_size=48),
            Text("||", font_size=48),
            Text("J1", font_size=48),
            Text("-", font_size=48),
            Text("J2", font_size=48),
            Text("||_2", font_size=48)
        ).arrange(RIGHT, buff=0.1)
        
        self.play(Transform(capri_word, formula))
        
        # Highlight components same color
        self.play(formula[4].animate.set_color(THEME_CYAN), formula[6].animate.set_color(THEME_YELLOW))
        
        # Slide 16: Zooms out and separates
        self.play(formula.animate.scale(0.6).to_edge(UP))
        self.wait(1)


# ---------------------------------------------------------------------------
# SCENE 3: The Math & Rejection (Slides 17-25)
# ---------------------------------------------------------------------------
class Scene3_Math(Scene):
    def construct(self):
        formula = VGroup(
            Text("p", font_size=48),
            Text("=", font_size=48),
            Text("SUM", font_size=48),
            Text("||", font_size=48),
            Text("J1", font_size=48),
            Text("-", font_size=48),
            Text("J2", font_size=48),
            Text("||_2", font_size=48)
        ).arrange(RIGHT, buff=0.1)
        formula[4].set_color(THEME_CYAN)
        formula[6].set_color(THEME_YELLOW)
        formula.scale(0.6).to_edge(UP)
        self.add(formula)

        # Slide 17: Pull back scene from intro
        cam1 = Rectangle(width=1, height=0.7, color=THEME_CYAN).shift(LEFT*3 + DOWN*1)
        cam2 = Rectangle(width=1, height=0.7, color=THEME_YELLOW).shift(RIGHT*3 + DOWN*1)
        bad_point = Dot(color=THEME_RED).shift(DOWN*1.5)
        
        line1 = Line(cam1.get_center(), bad_point, color=THEME_CYAN)
        line2 = Line(cam2.get_center(), bad_point, color=THEME_YELLOW)
        
        self.play(FadeIn(cam1, cam2, bad_point, line1, line2))

        # Slide 18 & 19: Highlight rho
        rho_box = SurroundingRectangle(formula[0], color=THEME_GREEN)
        self.play(Create(rho_box))
        
        rho_text = Text("Measures geometric deviation from reality", font_size=20, color=THEME_GREEN).next_to(rho_box, DOWN)
        self.play(Write(rho_text))
        
        # Slide 20, 21, 23: Lines warp for false match
        self.play(
            line1.animate.put_start_and_end_on(cam1.get_center(), bad_point.get_center() + RIGHT*0.5),
            line2.animate.put_start_and_end_on(cam2.get_center(), bad_point.get_center() + LEFT*0.5),
            bad_point.animate.set_opacity(0)
        )
        
        warp_text = Text("Lines of sight warp", color=THEME_RED, font_size=20).next_to(bad_point, DOWN)
        self.play(FadeIn(warp_text))

        # Slide 24: Does not equal zero
        not_eq = Text("!= 0", color=THEME_RED).scale(1.2).next_to(formula, RIGHT)
        self.play(Write(not_eq))

        # Slide 25: Rejected by CAPRI into bin
        reject_stamp = Text("REJECTED BY CAPRI", color=THEME_RED, font_size=40, weight=BOLD).rotate(0.2)
        self.play(FadeIn(reject_stamp, scale=2))
        
        # Bin
        bin_rect = VGroup(
            Line(LEFT*1, RIGHT*1),
            Line(LEFT*1, LEFT*0.8+DOWN*1.5),
            Line(RIGHT*1, RIGHT*0.8+DOWN*1.5),
            Line(LEFT*0.8+DOWN*1.5, RIGHT*0.8+DOWN*1.5)
        ).shift(DOWN*2.5 + RIGHT*4)
        bin_text = Text("Degenerates", font_size=16).next_to(bin_rect, DOWN)
        
        self.play(FadeOut(warp_text), FadeIn(bin_rect, bin_text))
        
        # Trash it
        trash_group = VGroup(line1, line2, reject_stamp)
        self.play(trash_group.animate.scale(0.2).move_to(bin_rect.get_center() + DOWN*0.5).set_opacity(0.5))
        self.wait(1)


# ---------------------------------------------------------------------------
# SCENE 4: Pipeline and Reconstruction (Slides 26-33)
# ---------------------------------------------------------------------------
class Scene4_Pipeline(Scene):
    def construct(self):
        # Slide 26 & 27: Wrap framework as CAPRI
        capri_box = Rectangle(width=3, height=2, color=THEME_GREEN)
        capri_text = Text("CAPRI", font_size=36, color=THEME_GREEN, weight=BOLD).move_to(capri_box)
        self.play(Create(capri_box), Write(capri_text))

        # Slide 28: Pipeline layout
        self.play(capri_box.animate.shift(ORIGIN), capri_text.animate.shift(ORIGIN))
        
        matcher_box = Rectangle(width=3, height=2, color=THEME_CYAN).shift(LEFT*4)
        matcher_text = Text("Feature\nMatcher", font_size=24).move_to(matcher_box)
        
        ba_box = Rectangle(width=3, height=2, color=THEME_YELLOW).shift(RIGHT*4)
        ba_text = Text("Bundle\nAdjustment", font_size=24).move_to(ba_box)
        
        arrow1 = Arrow(matcher_box.get_right(), capri_box.get_left())
        arrow2 = Arrow(capri_box.get_right(), ba_box.get_left())
        
        self.play(FadeIn(matcher_box, matcher_text, ba_box, ba_text, arrow1, arrow2))

        # Slide 30: Separates as points with colors
        dots = VGroup(*[Dot(color=THEME_YELLOW if i%2==0 else THEME_RED).shift(LEFT*4 + np.random.uniform(-0.5, 0.5, 3)) for i in range(10)])
        self.play(FadeOut(matcher_text), FadeIn(dots))

        # Slide 31: Dots move to CAPRI
        self.play(dots.animate.shift(RIGHT*4))
        
        # Slide 32: Left dots (Red) go to bin, right dots (Yellow) pass
        bin_rect = VGroup(
            Line(LEFT*0.5, RIGHT*0.5),
            Line(LEFT*0.5, LEFT*0.4+DOWN*1),
            Line(RIGHT*0.5, RIGHT*0.4+DOWN*1),
            Line(LEFT*0.4+DOWN*1, RIGHT*0.4+DOWN*1)
        ).shift(DOWN*3)
        bin_text = Text("Degenerates", font_size=16).next_to(bin_rect, DOWN)
        self.play(FadeIn(bin_rect, bin_text))

        red_dots = VGroup(*[dots[i] for i in range(10) if i % 2 != 0])
        yellow_dots = VGroup(*[dots[i] for i in range(10) if i % 2 == 0])
        
        self.play(
            red_dots.animate.move_to(bin_rect.get_center()).set_opacity(0.3),
            yellow_dots.animate.shift(RIGHT*4)
        )

        # Slide 33: Bundle adjustment expands into 3D model
        self.play(FadeOut(ba_text))
        
        # Simulate 3D model with a mesh/grid of dots
        model_dots = VGroup(*[Dot(color=THEME_YELLOW, radius=0.03).move_to(ba_box.get_center() + np.array([x*0.2, y*0.2, 0])) 
                              for x in range(-4, 5) for y in range(-4, 5) if x**2 + y**2 < 16])
        
        self.play(Transform(yellow_dots, model_dots))
        
        # Rotate model to look 3D
        self.play(model_dots.animate.stretch(0.5, 0).rotate(PI/4), run_time=2)
        self.wait(2)
