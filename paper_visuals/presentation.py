from manim import *
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
import os

# To run this, you will need to install manim: `pip install manim`
# Then run: `manim -pql presentation.py` (to render all scenes at low quality for quick preview)
# Or render a specific scene: `manim -pqh presentation.py Scene1_Problem`

# ---------------------------------------------------------------------------
# Section 1: The Problem (Standard SfM and Degeneracy)
# ---------------------------------------------------------------------------
class Scene1_Problem(Scene):
    def construct(self):
        # Slide 1: Scattered photos of a single object
        title = Text("Standard Structure-from-Motion (SfM)").to_edge(UP)
        self.play(Write(title))

        # Using placeholder paths for the real images you specified
        # You can replace 'placeholder_img.jpg' with actual filenames in those dirs
        img_path_1 = str(PROJECT_ROOT / r"data/real_images/image-matching-challenge-2025/sample1.jpg")
        img_path_2 = str(PROJECT_ROOT / r"data/real_images/sample2.jpg")
        
        # Fallback rectangles if images don't exist during render
        cam1 = Rectangle(width=2, height=1.5, color=BLUE).shift(LEFT*3 + DOWN*1)
        cam2 = Rectangle(width=2, height=1.5, color=BLUE).shift(RIGHT*3 + DOWN*1)
        cam3 = Rectangle(width=2, height=1.5, color=BLUE).shift(UP*2)

        self.play(FadeIn(cam1), FadeIn(cam2), FadeIn(cam3))
        self.wait(1)

        # Slide 2: Lines and feature matching
        point_3d = Dot(color=YELLOW).shift(ORIGIN)
        line1 = DashedLine(cam1.get_center(), point_3d, color=WHITE)
        line2 = DashedLine(cam2.get_center(), point_3d, color=WHITE)
        line3 = DashedLine(cam3.get_center(), point_3d, color=WHITE)

        self.play(Create(line1), Create(line2), Create(line3))
        self.play(FadeIn(point_3d))
        self.wait(1)

class Scene2_Degeneracy(Scene):
    def construct(self):
        # Slide 3: Zoom into degeneracy
        title = Text("The Degeneracy Problem").to_edge(UP)
        self.add(title)

        bad_point = Dot(color=RED).shift(DOWN*2)
        cam1 = Dot().shift(LEFT*3 + UP*1)
        cam2 = Dot().shift(RIGHT*3 + UP*1)

        line_bad1 = Line(cam1.get_center(), bad_point, color=RED)
        line_bad2 = Line(cam2.get_center(), bad_point, color=RED)

        self.play(Create(line_bad1), Create(line_bad2))
        self.play(FadeIn(bad_point))
        
        warning = Text("Reflection / False Match", color=RED).next_to(bad_point, DOWN)
        self.play(Write(warning))
        self.wait(2)

# ---------------------------------------------------------------------------
# Section 2: Introduction to CAPRI
# ---------------------------------------------------------------------------
class Scene3_IntroCAPRI(Scene):
    def construct(self):
        # Slide 5: Title
        title_main = Text("CAPRI", font_size=72, weight=BOLD, color=BLUE)
        subtitle = Text("Conditioning-Aware Projective Invariants\nfor Calibration-Free 3D Reconstruction", font_size=24)
        author = Text("Kurt Andrei L. Gabriel", font_size=24, color=GRAY)
        
        VGroup(title_main, subtitle, author).arrange(DOWN, buff=0.5)

        self.play(Write(title_main))
        self.play(FadeIn(subtitle))
        self.play(FadeIn(author))
        self.wait(2)
        
        self.play(FadeOut(subtitle), FadeOut(author), title_main.animate.to_edge(UP).scale(0.5))

        # Slide 6 & 7: Objectives and Contributions (Visual)
        box = Rectangle(width=6, height=3, color=GREEN, fill_opacity=0.1)
        box_text = Text("Strict Algebraic Filter", font_size=36)
        filter_group = VGroup(box, box_text).center()

        self.play(Create(box), Write(box_text))
        
        desc = Text("Ensures only well-posed, geometrically\nconsistent components enter the reconstruction.", font_size=24).next_to(box, DOWN)
        self.play(FadeIn(desc))
        self.wait(2)

# ---------------------------------------------------------------------------
# Section 3: The Maths (3B1B Style)
# ---------------------------------------------------------------------------
class Scene4_Math(Scene):
    def construct(self):
        # Slide 8: The Formula
        # Using Six-point bracket-ratio invariants and consistency residual rho
        title = Text("Enforcing Geometric Laws", font_size=36).to_edge(UP)
        self.add(title)

        # 3B1B Style equation splitting
        # rho = || J^(1) - J^(2) ||_2
        eq_full = MathTex(r"\rho", r"=", r"\sum", r"\left\|", r"\mathbf{J}^{(1)}_i", r"-", r"\mathbf{J}^{(2)}_i", r"\right\|_2")
        eq_full.scale(1.5)

        self.play(Write(eq_full))
        self.wait(1)

        # Slide 9-n: Break it down
        # Highlight term 1 (View 1 invariant)
        term1_desc = Text("Projective Invariant (View 1)", font_size=24, color=BLUE).next_to(eq_full, DOWN*2)
        self.play(eq_full[4].animate.set_color(BLUE), FadeIn(term1_desc))
        self.wait(1)

        # Highlight term 2 (View 2 invariant)
        term2_desc = Text("Projective Invariant (View 2)", font_size=24, color=YELLOW).next_to(eq_full, DOWN*2)
        self.play(FadeOut(term1_desc), eq_full[6].animate.set_color(YELLOW), FadeIn(term2_desc))
        self.wait(1)
        self.play(FadeOut(term2_desc))

        # The rejection animation
        rejection_text = Text("If heavily deformed by degeneracy...", font_size=24).next_to(eq_full, UP*2)
        self.play(FadeIn(rejection_text))
        
        not_equal = MathTex(r"\neq", r"0", color=RED).scale(1.5).next_to(eq_full, RIGHT)
        self.play(Write(not_equal))
        
        cross = Cross(eq_full, stroke_color=RED, stroke_width=6)
        self.play(Create(cross))
        self.wait(2)

# ---------------------------------------------------------------------------
# Section 4: Computational Implementation
# ---------------------------------------------------------------------------
class Scene5_Pipeline(Scene):
    def construct(self):
        title = Text("Where CAPRI fits").to_edge(UP)
        self.play(Write(title))

        # Pipeline Flowchart
        box1 = Text("Feature\nMatching", font_size=24).shift(LEFT*4)
        box2 = Text("CAPRI\nLayer", font_size=24, color=GREEN, weight=BOLD).shift(ORIGIN)
        box3 = Text("Bundle\nAdjustment", font_size=24).shift(RIGHT*4)
        
        r1 = SurroundingRectangle(box1, color=WHITE)
        r2 = SurroundingRectangle(box2, color=GREEN, fill_opacity=0.2)
        r3 = SurroundingRectangle(box3, color=WHITE)

        a1 = Arrow(box1.get_right(), box2.get_left())
        a2 = Arrow(box2.get_right(), box3.get_left())

        self.play(FadeIn(box1, r1))
        self.play(GrowArrow(a1))
        self.play(FadeIn(box2, r2))
        self.play(GrowArrow(a2))
        self.play(FadeIn(box3, r3))
        
        # Show filtering
        bad_points = VGroup(*[Dot(color=RED).shift(LEFT*2 + UP*np.random.uniform(-1, 1)) for _ in range(5)])
        good_points = VGroup(*[Dot(color=YELLOW).shift(LEFT*2 + UP*np.random.uniform(-1, 1)) for _ in range(5)])
        
        self.play(FadeIn(bad_points), FadeIn(good_points))
        
        # Good points pass
        self.play(good_points.animate.shift(RIGHT*4))
        # Bad points fall away
        self.play(bad_points.animate.shift(DOWN*3 + LEFT*1).set_opacity(0))
        
        self.wait(2)

# ---------------------------------------------------------------------------
# Section 5 & 6: Experiments and Conclusion
# ---------------------------------------------------------------------------
class Scene6_Conclusion(Scene):
    def construct(self):
        title = Text("Experimental Validation & Impact").to_edge(UP)
        self.play(Write(title))

        # You can load real images as ImageMobject here
        # E.g., img = ImageMobject(str(PROJECT_ROOT / r"dust3r/assets/sample.png"))
        
        chart_mockup = VGroup(
            Line(DOWN*2+LEFT*3, DOWN*2+RIGHT*3),
            Line(DOWN*2+LEFT*3, UP*2+LEFT*3),
            Rectangle(width=1, height=3, fill_color=BLUE, fill_opacity=0.8).shift(LEFT*1.5 + DOWN*0.5),
            Rectangle(width=1, height=1, fill_color=GREEN, fill_opacity=0.8).shift(RIGHT*1.5 + DOWN*1.5),
        )
        labels = VGroup(
            Text("Baseline Error", font_size=20).next_to(chart_mockup[2], UP),
            Text("CAPRI Error", font_size=20).next_to(chart_mockup[3], UP),
        )

        self.play(Create(chart_mockup[0]), Create(chart_mockup[1]))
        self.play(GrowFromBottom(chart_mockup[2]), GrowFromBottom(chart_mockup[3]))
        self.play(Write(labels))
        self.wait(1)
        
        self.play(FadeOut(chart_mockup), FadeOut(labels))
        
        # Conclusion text
        conc1 = Text("1. Radically improved geometric stability", font_size=32)
        conc2 = Text("2. Zero reliance on camera intrinsics", font_size=32)
        conc3 = Text("3. Path to robust autonomous navigation", font_size=32)
        
        conc_group = VGroup(conc1, conc2, conc3).arrange(DOWN, aligned_edge=LEFT).shift(DOWN*0.5)
        
        self.play(FadeIn(conc1))
        self.play(FadeIn(conc2))
        self.play(FadeIn(conc3))
        
        self.wait(3)
