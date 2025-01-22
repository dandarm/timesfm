#!/usr/bin/env bash
#
# Script di esempio per costruire Python "user-only" con pyenv,
# compilando le librerie necessarie (ncurses, readline, libffi, ecc.)
# in ~/local e linkando con rpath. Funziona su macchine dove NON hai sudo.

set -e   # Se un comando va in errore, lo script si interrompe

##################################
# 1) CONFIGURAZIONE DI BASE
##################################

# Dove installiamo le librerie locali:
INSTALL_DIR="/workspace/local"

# Quale versione di Python vuoi installare con pyenv:
PYTHON_VERSION="3.10.16"

# Se vuoi partire proprio da zero, cancella la dir (ATTENZIONE: rimuove tutto)
# rm -rf "$INSTALL_DIR"

mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"
echo "Installazione librerie in: $INSTALL_DIR"

##################################
# 2) COMPILAZIONE DI NCURSES
##################################
# Scarica l'archivio (versione di esempio, vedi ftp.gnu.org/gnu/ncurses)
NCURSES_VERSION="6.3"
wget https://ftp.gnu.org/gnu/ncurses/ncurses-$NCURSES_VERSION.tar.gz
tar xvf ncurses-$NCURSES_VERSION.tar.gz
cd ncurses-$NCURSES_VERSION

# --enable-widec produce libncursesw.so, supporto Unicode. 
# --with-shared crea la .so. 
# Usiamo -Wl,-rpath per incorporare ~/local/lib come "percorso di runtime".
./configure --prefix="$INSTALL_DIR" --with-shared --enable-widec \
    LDFLAGS="-Wl,-rpath=$INSTALL_DIR/lib"

make -j4
make install
cd ..

##################################
# 3) COMPILAZIONE DI READLINE
##################################
# Scarica l'archivio (versione di esempio)
RL_VERSION="8.2"
wget https://ftp.gnu.org/gnu/readline/readline-$RL_VERSION.tar.gz
tar xvf readline-$RL_VERSION.tar.gz
cd readline-$RL_VERSION

# Forziamo la linkage con -lncursesw. 
# Se "tinfo" è separata, aggiungere anche -ltinfow se serve.
./configure --prefix="$INSTALL_DIR" \
    CFLAGS="-I$INSTALL_DIR/include" \
    LDFLAGS="-L$INSTALL_DIR/lib -Wl,-rpath=$INSTALL_DIR/lib"
make SHLIB_LIBS="-lncursesw" -j4
make install
cd ..

##################################
# 4) COMPILAZIONE DI LIBFFI
##################################
# Scarica (esempio: libffi-3.4.4)
FFI_VERSION="3.4.4"
wget https://github.com/libffi/libffi/releases/download/v$FFI_VERSION/libffi-$FFI_VERSION.tar.gz
tar xvf libffi-$FFI_VERSION.tar.gz
cd libffi-$FFI_VERSION

# libffi a volte mette gli header in $INSTALL_DIR/lib/libffi-<version>/include
./configure --prefix="$INSTALL_DIR" LDFLAGS="-Wl,-rpath=$INSTALL_DIR/lib"
make -j4
make install
cd ..

##################################
# (OPZIONALE) ALTRE LIBRERIE
##################################
# Se servono bzip2, xz, sqlite, openssl, ecc. stessa logica:
#   - ./configure --prefix="$INSTALL_DIR" LDFLAGS="-Wl,-rpath=$INSTALL_DIR/lib"
#   - make -j4
#   - make install


##### openssl
wget https://www.openssl.org/source/openssl-3.1.2.tar.gz
tar xvf openssl-3.1.2.tar.gz
cd openssl-3.1.2
./Configure --prefix="$INSTALL_DIR" LDFLAGS="-Wl,-rpath=$INSTALL_DIR/lib" --openssldir="$INSTALL_DIR/openssl"
make -j4
make install_sw
cd ..

### bz2
wget https://sourceware.org/pub/bzip2/bzip2-1.0.8.tar.gz
tar xvf bzip2-1.0.8.tar.gz
cd bzip2-1.0.8
make -f Makefile-libbz2_so
make -j4
make install PREFIX="$INSTALL_DIR"


SQLITE_VER="3420000"
wget https://www.sqlite.org/2023/sqlite-autoconf-$SQLITE_VER.tar.gz
tar xvf sqlite-autoconf-$SQLITE_VER.tar.gz
cd sqlite-autoconf-$SQLITE_VER

./configure --prefix="$INSTALL_DIR" LDFLAGS="-Wl,-rpath=$INSTALL_DIR/lib64 -L$INSTALL_DIR/lib64"

make -j4
make install
cd ..



XZ_VER="5.4.3"
wget https://tukaani.org/xz/xz-$XZ_VER.tar.gz
tar xvf xz-$XZ_VER.tar.gz
cd xz-$XZ_VER

./configure --prefix="$INSTALL_DIR" LDFLAGS="-Wl,-rpath=$INSTALL_DIR/lib64 -L$INSTALL_DIR/lib64"

make -j4
make install

cd ..



#########################################à


curl https://pyenv.run | bash
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init - bash)"
eval "$(pyenv virtualenv-init -)"


curl -sSL https://install.python-poetry.org | python3 -
export PATH="$HOME/.local/bin:$PATH"

##################################
# 5) INSTALLAZIONE PYTHON CON PYENV
##################################
# Prima di lanciare pyenv install, settiamo le variabili in questa shell.
unset LD_LIBRARY_PATH  # Evita conflitti con awk e tool di sistema
export CPPFLAGS="-I$INSTALL_DIR/include -I$INSTALL_DIR/include/ncurses -I$INSTALL_DIR/include/ncursesw -I$INSTALL_DIR/openssl-3.1.2/include -I$INSTALL_DIR/lib64"
export LDFLAGS="-L$INSTALL_DIR/lib -L$INSTALL_DIR/lib64 -L$INSTALL_DIR/openssl-3.1.2 -Wl,-rpath=$INSTALL_DIR/lib -Wl,-rpath=$INSTALL_DIR/lib64"
# Se libffi ha messo gli .h in una sottodir, aggiungi:
# export CPPFLAGS="$CPPFLAGS -I$INSTALL_DIR/lib/libffi-$FFI_VERSION/include"
export PKG_CONFIG_PATH="$INSTALL_DIR/lib/pkgconfig:$PKG_CONFIG_PATH"
export LD_LIBRARY_PATH="$INSTALL_DIR/lib64:$LD_LIBRARY_PATH"


echo "Compilo Python $PYTHON_VERSION con pyenv..."
# Se non hai ancora pyenv installato, dovresti averlo configurato (ad es. in ~/.bashrc).
# Pyenv deve essere nel PATH, e la shell deve "vederlo":
export CONFIGURE_OPTS="--with-openssl=$INSTALL_DIR --with-openssl-rpath=auto"
pyenv install -v $PYTHON_VERSION

echo "Python $PYTHON_VERSION installato con pyenv in ~/.pyenv/versions/$PYTHON_VERSION"

##################################
# 6) TEST FINALE
##################################
echo "Test moduli..."
pyenv shell $PYTHON_VERSION
python -c "import sys; print(sys.version)"
python -c "import curses, curses.panel, ctypes, readline; print('OK, tutto a posto!')"

echo "Installazione completata."

# Fine script
