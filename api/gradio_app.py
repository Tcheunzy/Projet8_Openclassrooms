"""Interface Gradio de démonstration.

Cette interface est un *client HTTP* de l'API : elle ne charge ni le modèle
ni le pipeline. Tout ce qu'elle affiche provient d'une réponse de /predict,
ce qui garantit qu'il n'existe qu'un seul chemin de prédiction.
"""
import os

import gradio as gr
import httpx

API_URL = os.getenv("API_URL", f"http://localhost:{os.getenv('PORT', '8000')}")

# L'interface est montée dans le même processus que l'API, mais elle l'appelle
# par HTTP comme n'importe quel client : elle doit donc s'authentifier comme
# n'importe quel client. C'est la contrepartie du choix d'en faire un client
# et non un appelant direct du pipeline.
API_KEY = os.getenv("API_KEY")

# Doit correspondre exactement au Literal de api/schemas.py
EDUCATION = [
    "Lower secondary",
    "Secondary / secondary special",
    "Incomplete higher",
    "Higher education",
    "Academic degree",
]


def entetes() -> dict:
    """En-têtes de la requête, avec la clé d'API si elle est configurée."""
    valeurs = {"Content-Type": "application/json"}
    if API_KEY:
        valeurs["X-API-Key"] = API_KEY
    return valeurs


def predire(sk_id_curr, age, anciennete, genre, contrat, education,
            possede_voiture, possede_bien, nb_enfants, nb_personnes,
            revenu, montant_credit, mensualite, prix_bien,
            ext_1, ext_2, ext_3):
    """Traduit la saisie humaine en requête API, puis met en forme la réponse."""

    payload = {
        "SK_ID_CURR": int(sk_id_curr),
        # L'utilisateur saisit des années ; le modèle attend des jours négatifs
        "DAYS_BIRTH": -int(age * 365.25),
        "DAYS_EMPLOYED": -int(anciennete * 365.25),
        "CODE_GENDER": genre,
        "NAME_CONTRACT_TYPE": contrat,
        "NAME_EDUCATION_TYPE": education,
        "FLAG_OWN_CAR": possede_voiture,
        "FLAG_OWN_REALTY": possede_bien,
        "CNT_CHILDREN": int(nb_enfants),
        "CNT_FAM_MEMBERS": float(nb_personnes),
        "AMT_INCOME_TOTAL": float(revenu),
        "AMT_CREDIT": float(montant_credit),
        "AMT_ANNUITY": float(mensualite),
        "AMT_GOODS_PRICE": float(prix_bien),
    }

    # Les scores externes sont facultatifs : un champ vidé devient None,
    # et l'imputer du préprocesseur prendra le relais côté API.
    for nom, valeur in [("EXT_SOURCE_1", ext_1),
                        ("EXT_SOURCE_2", ext_2),
                        ("EXT_SOURCE_3", ext_3)]:
        if valeur is not None:
            payload[nom] = float(valeur)

    try:
        response = httpx.post(f"{API_URL}/predict", json=payload,
                              headers=entetes(), timeout=30.0)
    except httpx.RequestError as exc:
        return f"### Erreur de connexion à l'API\n\n`{exc}`"

    if response.status_code == 401:
        return ("### Accès refusé\n\n"
                "L'API exige une clé d'authentification. Renseignez la "
                "variable d'environnement `API_KEY` du service.")

    if response.status_code == 422:
        details = response.json()["detail"]
        lignes = "\n".join(
            f"- `{'.'.join(str(p) for p in d['loc'][1:])}` : {d['msg']}"
            for d in details
        )
        return f"### Saisie invalide\n\n{lignes}"

    if response.status_code != 200:
        return f"### Erreur {response.status_code}\n\n`{response.text}`"

    body = response.json()
    couleur = "🔴" if body["decision"] == "refusé" else "🟢"
    return (
        f"## {couleur} Crédit {body['decision']}\n\n"
        f"**Probabilité de défaut : {body['probability']:.1%}**\n\n"
        f"Seuil de décision : {body['threshold']:.0%} — "
        f"au-delà, le dossier est refusé.\n\n"
        f"*Client {body['sk_id_curr']} • modèle version "
        f"{body['mlflow_model_version']}*"
    )


def build_demo():
    """Construit l'interface. Appelée par api/main.py au moment du montage."""
    with gr.Blocks(title="Scoring crédit — Prêt à dépenser") as demo:
        gr.Markdown(
            "# Prêt à dépenser — Évaluation de dossier\n"
            "Renseignez les informations du client, puis lancez l'évaluation. "
            "Les champs non saisis sont estimés par le modèle."
        )

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Identité")
                sk_id = gr.Number(label="Identifiant client", value=100002, precision=0)
                age = gr.Slider(18, 70, value=44, step=1, label="Âge (années)")
                anciennete = gr.Slider(0, 45, value=6, step=1,
                                       label="Ancienneté dans l'emploi (années)")
                genre = gr.Radio(["F", "M"], value="F", label="Genre")
                education = gr.Dropdown(EDUCATION, value="Higher education",
                                        label="Niveau d'études")
                nb_enfants = gr.Number(label="Nombre d'enfants", value=0, precision=0)
                nb_personnes = gr.Number(label="Personnes au foyer", value=2, precision=0)

            with gr.Column():
                gr.Markdown("### Dossier de crédit")
                contrat = gr.Radio(["Cash loans", "Revolving loans"],
                                   value="Cash loans", label="Type de crédit")
                revenu = gr.Number(label="Revenu annuel (€)", value=150000)
                montant_credit = gr.Number(label="Montant du crédit (€)", value=500000)
                mensualite = gr.Number(label="Mensualité (€)", value=25000)
                prix_bien = gr.Number(label="Prix du bien financé (€)", value=450000)
                possede_voiture = gr.Radio(["Y", "N"], value="Y",
                                           label="Possède un véhicule")
                possede_bien = gr.Radio(["Y", "N"], value="Y",
                                        label="Possède un bien immobilier")

        gr.Markdown("### Scores externes *(facultatifs — videz le champ si inconnu)*")
        with gr.Row():
            ext_1 = gr.Number(label="EXT_SOURCE_1", value=0, minimum=0, maximum=1)
            ext_2 = gr.Number(label="EXT_SOURCE_2", value=0, minimum=0, maximum=1)
            ext_3 = gr.Number(label="EXT_SOURCE_3", value=0, minimum=0, maximum=1)

        bouton = gr.Button("Évaluer le dossier", variant="primary")
        resultat = gr.Markdown()

        bouton.click(
            fn=predire,
            inputs=[sk_id, age, anciennete, genre, contrat, education,
                    possede_voiture, possede_bien, nb_enfants, nb_personnes,
                    revenu, montant_credit, mensualite, prix_bien,
                    ext_1, ext_2, ext_3],
            outputs=resultat,
        )

    return demo


if __name__ == "__main__":
    build_demo().launch()