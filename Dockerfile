# [1] base setting
FROM nvidia/cuda:10.1-cudnn7-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV HOME /root

# [2] zsh
RUN apt-get update && apt-get -y upgrade && \
    apt-get install -y \
    wget \
    curl \
    git \
    vim-athena \ 
    zsh \
    unzip --no-install-recommends

SHELL ["/bin/zsh", "-c"]
RUN wget http://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh -O - | zsh


# [3] pyenv

RUN apt-get update && \
    apt-get install -y \
    make \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
    python-openssl --no-install-recommends


RUN curl https://pyenv.run | zsh && \
    echo '' >> $HOME/.zshrc && \
    echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> $HOME/.zshrc && \
    echo 'eval "$(pyenv init -)"' >> $HOME/.zshrc && \
    echo 'eval "$(pyenv virtualenv-init -)"' >> $HOME/.zshrc

ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

RUN pyenv install 3.7.4 && \
    pyenv global 3.7.4 && \
    pyenv rehash


# [4] python
RUN apt-get update && apt-get install -y ffmpeg nodejs npm
RUN pip install setuptools && \
    pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html && \
    echo 'alias jl="DISPLAY=:0 jupyter lab --ip 0.0.0.0 --port 8888 --allow-root &"' >> /root/.zshrc && \
    echo 'alias tb="tensorboard --logdir runs --bind_all &"' >> $HOME/.zshrc

# [5] Disentanglement
ENV DISENTANGLEMENT_LIB_DATA /share2/iwasawa/dataset

# [x] dotfiles
WORKDIR $HOME
RUN git clone https://github.com/yusuke0519/dotfiles.git
WORKDIR $HOME/dotfiles
RUN sh setup.sh
WORKDIR $HOME/.vim/bundle/neobundle.vim/bin
RUN sh neoinstall

RUN apt-get clean \
    && rm -rf /var/lib/apt/lists/*

CMD ["/bin/zsh"]
