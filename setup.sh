#!/usr/bin/env bash
# =============================================================================
# setup.sh — Configuración completa del dashboard SOL/USD
# Crea el repo en GitHub, pushea todos los archivos y deploya en Vercel.
#
# REQUISITOS (instalar antes de correr):
#   - gh (GitHub CLI):  https://cli.github.com/
#   - vercel (CLI):     npm install -g vercel
#   - git
#
# USO:
#   chmod +x setup.sh
#   ./setup.sh
# =============================================================================

set -e  # Salir si cualquier comando falla

# ── Colores ──────────────────────────────────────────────────────────────────
BLUE='\033[0;34m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[OK]${NC}   $1"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
error()   { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

echo ""
echo "=================================================="
echo "   🔷  SOL/USD Dashboard — Setup automático"
echo "=================================================="
echo ""

# ── 0. Verificar herramientas ─────────────────────────────────────────────────
info "Verificando herramientas..."
command -v git    >/dev/null 2>&1 || error "git no instalado. Instalalo desde https://git-scm.com"
command -v gh     >/dev/null 2>&1 || error "GitHub CLI (gh) no instalado. Instalalo desde https://cli.github.com"
command -v vercel >/dev/null 2>&1 || error "Vercel CLI no instalado. Corré: npm install -g vercel"
success "git, gh y vercel están disponibles."

# ── 1. Autenticación ──────────────────────────────────────────────────────────
info "Verificando sesión de GitHub..."
if ! gh auth status >/dev/null 2>&1; then
  warn "No estás logueado en GitHub CLI. Iniciando login..."
  gh auth login
fi
success "GitHub autenticado."

info "Verificando sesión de Vercel..."
if ! vercel whoami >/dev/null 2>&1; then
  warn "No estás logueado en Vercel CLI. Iniciando login..."
  vercel login
fi
success "Vercel autenticado."

# ── 2. Pedir nombre de repo ───────────────────────────────────────────────────
echo ""
read -p "$(echo -e ${YELLOW})Nombre del repositorio GitHub [sol-sentiment-dashboard]: $(echo -e ${NC})" REPO_NAME
REPO_NAME="${REPO_NAME:-sol-sentiment-dashboard}"
read -p "$(echo -e ${YELLOW})¿Repositorio público o privado? (public/private) [public]: $(echo -e ${NC})" REPO_VIS
REPO_VIS="${REPO_VIS:-public}"

# ── 3. Crear repositorio GitHub ───────────────────────────────────────────────
echo ""
info "Creando repositorio GitHub: $REPO_NAME ($REPO_VIS)..."

GH_USER=$(gh api user --jq '.login')
REPO_URL="https://github.com/$GH_USER/$REPO_NAME"

if gh repo view "$GH_USER/$REPO_NAME" >/dev/null 2>&1; then
  warn "El repositorio $REPO_NAME ya existe. Usándolo."
else
  gh repo create "$REPO_NAME" --$REPO_VIS --description "Dashboard SOL/USD con análisis de sentimiento Reddit - Tesina LCC Datos"
  success "Repositorio creado: $REPO_URL"
fi

# ── 4. Git init y primer commit ───────────────────────────────────────────────
info "Inicializando git y realizando primer commit..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d ".git" ]; then
  git init
  git branch -m main
fi

# Crear .gitignore si no existe
if [ ! -f ".gitignore" ]; then
cat > .gitignore << 'GITIGNORE'
# Python
__pycache__/
*.pyc
*.pyo
.env
venv/
*.egg-info/

# Node
dashboard/node_modules/
dashboard/.next/
dashboard/.env.local

# Sistema
.DS_Store
*.log
GITIGNORE
  success ".gitignore creado."
fi

git add -A
git commit -m "feat: initial commit — SOL/USD sentiment dashboard" 2>/dev/null || warn "Nada nuevo para commitear."

# ── 5. Conectar remote y push ─────────────────────────────────────────────────
info "Conectando remote a GitHub..."
REMOTE_URL="https://github.com/$GH_USER/$REPO_NAME.git"

if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "$REMOTE_URL"
else
  git remote add origin "$REMOTE_URL"
fi

info "Pusheando a GitHub..."
git push -u origin main
success "Código subido a $REPO_URL"

# ── 6. Configurar Secrets de GitHub ──────────────────────────────────────────
echo ""
echo "──────────────────────────────────────────────────"
echo "  Configuración de Secrets (para el cron diario)"
echo "──────────────────────────────────────────────────"
warn "Necesitás 4 secrets de GitHub para que el cron funcione."
warn "Podés configurarlos ahora o hacerlo después en: $REPO_URL/settings/secrets/actions"
echo ""
read -p "$(echo -e ${YELLOW})¿Configurar secrets ahora? (s/n) [s]: $(echo -e ${NC})" SETUP_SECRETS
SETUP_SECRETS="${SETUP_SECRETS:-s}"

if [[ "$SETUP_SECRETS" == "s" || "$SETUP_SECRETS" == "S" ]]; then
  echo ""
  read -p "  COINGECKO_API_KEY    : " CG_KEY
  read -p "  REDDIT_CLIENT_ID     : " RD_ID
  read -p "  REDDIT_CLIENT_SECRET : " RD_SECRET
  read -p "  REDDIT_USER_AGENT    : " RD_UA

  gh secret set COINGECKO_API_KEY    -b "$CG_KEY"    --repo "$GH_USER/$REPO_NAME"
  gh secret set REDDIT_CLIENT_ID     -b "$RD_ID"     --repo "$GH_USER/$REPO_NAME"
  gh secret set REDDIT_CLIENT_SECRET -b "$RD_SECRET" --repo "$GH_USER/$REPO_NAME"
  gh secret set REDDIT_USER_AGENT    -b "$RD_UA"     --repo "$GH_USER/$REPO_NAME"
  success "Secrets configurados correctamente."
else
  warn "Recordá configurar los secrets manualmente en: $REPO_URL/settings/secrets/actions"
fi

# ── 7. Deploy en Vercel ───────────────────────────────────────────────────────
echo ""
info "Deployando dashboard en Vercel..."
cd "$SCRIPT_DIR/dashboard"

# Link y deploy del proyecto
vercel link --yes --project="$REPO_NAME"
vercel --prod --yes

DEPLOYMENT_URL=$(vercel ls "$REPO_NAME" --json 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d[0]['url'])" 2>/dev/null || echo "Ver en https://vercel.com/dashboard")

echo ""
echo "============================================"
success "¡Todo listo! 🚀"
echo ""
echo "  📦 GitHub:  $REPO_URL"
echo "  🌐 Vercel:  https://$DEPLOYMENT_URL"
echo ""
echo "  ⏰ El cron corre todos los días a las 06:00 UTC (03:00 ART)"
echo "     y actualiza los datos automáticamente."
echo "============================================"
echo ""
