#!/usr/bin/env python3
import aws_cdk as cdk

from image_generation_app.image_generation_app_stack import ImageGenerationAppStack

app = cdk.App()
ImageGenerationAppStack(app, "PyImageApiStack")
app.synth()
