"""
Visita o app no Streamlit Community Cloud com um navegador real (Playwright),
espera o JavaScript carregar e a conexão WebSocket abrir, e clica no botão
"Yes, get this app back up!" se o app estiver dormindo.

Um simples GET (curl/requests) NÃO funciona aqui: a resposta é só uma casca
HTML estática — o app Python só liga depois que o JS roda no navegador.
"""

from playwright.sync_api import sync_playwright

APP_URL = "https://analise-blue-sky.streamlit.app/"


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        print(f"Visitando {APP_URL} ...")
        page.goto(APP_URL, timeout=60000)

        # Dá tempo para o JS carregar e a conexão WebSocket com o backend abrir
        page.wait_for_timeout(6000)

        # Se o app estiver dormindo, aparece um botão "Yes, get this app back up!"
        botao = page.get_by_text("get this app back up", exact=False)
        if botao.count() > 0 and botao.first.is_visible():
            print("App estava dormindo — clicando para acordar...")
            botao.first.click()
            # Espera o app subir de fato antes de encerrar
            page.wait_for_timeout(20000)
            print("Comando de despertar enviado.")
        else:
            print("App já estava acordado — nada a fazer.")

        browser.close()


if __name__ == "__main__":
    main()
