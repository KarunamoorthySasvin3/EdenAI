import Document, { Html, Head, Main, NextScript } from "next/document";

class MyDocument extends Document {
  render() {
    return (
      <Html lang="en" className="light" style={{ colorScheme: "light" }}>
        <Head />
        <body>
          <Main />
          <NextScript />
          <script
            dangerouslySetInnerHTML={{
              __html: `
                document.addEventListener("DOMContentLoaded", function() {
                  // Remove unexpected attributes injected by browser extensions (e.g. Grammarly)
                  document.documentElement.removeAttribute('data-lt-installed');
                  document.body.removeAttribute('data-new-gr-c-s-check-loaded');
                  document.body.removeAttribute('data-gr-ext-installed');
                });
              `,
            }}
          />
        </body>
      </Html>
    );
  }
}

export default MyDocument;
