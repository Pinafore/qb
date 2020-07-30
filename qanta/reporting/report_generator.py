from jinja2 import Environment, PackageLoader
from qanta import qlogging


log = qlogging.get(__name__)


class ReportGenerator:
    def __init__(self, template):
        self.template = template

    def create(self, variables, md_output, pdf_output):
        env = Environment(loader=PackageLoader("qanta", "reporting/templates"))
        template = env.get_template(self.template)
        markdown = template.render(variables)
        if md_output is not None:
            with open(md_output, "w") as f:
                f.write(markdown)
        try:
            import pypandoc

            pypandoc.convert_text(
                markdown,
                "pdf",
                format="md",
                outputfile=pdf_output,
                extra_args=["-V", "geometry:margin=.75in"],
            )
        except Exception as e:
            log.warn(
                "Pandoc was not installed or there was an error calling it, omitting PDF report"
            )
            log.warn(str(e))
