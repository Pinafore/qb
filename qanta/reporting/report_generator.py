from jinja2 import Environment, PackageLoader
import pypandoc


class ReportGenerator:
    def __init__(self, template):
        self.template = template

    def create(self, variables, output):
        env = Environment(loader=PackageLoader('qanta', 'reporting/templates'))
        template = env.get_template(self.template)
        markdown = template.render(variables)
        pypandoc.convert_text(
            markdown,
            'pdf',
            format='md',
            outputfile=output,
            extra_args=['-V', 'geometry:margin=.75in']
        )
