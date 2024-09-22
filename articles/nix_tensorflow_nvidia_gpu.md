> [!NOTE] TLDR
> This is a simple explanation on how to use **nix** to solve some very annoying  problems regarding the setup of **TensorFlow** in local machines with **cuda support**.
> If you just need a solution, skip ahead and take a look at my system config file (`config.nix`) an the nix-shell code on (`shell.nix`).

# First things first: Fuck you NVIDIA!
![First things first](./_data/f_u_nvidia.gif)

Its widely know that NVIDIA support for Linux is and has always been shit. As a long time Linux user, I remember always having trouble since my early days on Ubuntu 16 or something like this (keep in my I'm fairly young).

From broken systems due to strange errors, to not being able to use formally supported features, everything with NVIDIA products on Linux seems to be the flip of a coin:
- **Heads: you're fucked, nothing works.**
- **Tails: it kind of works**, and maybe you'll be able to get some work done before an update breaks some part of the gigantic castle of cards that is the NVIDIA ecosystem, **and then you're fucked**.

It has been always like that for me.

## The house special

I use local machines to do some work regarding [machine learning (ML)](https://en.wikipedia.org/wiki/Machine_learning) models, academic work with [physicochemical simulations](https://en.wikipedia.org/wiki/Process_simulation) and play some games. Obviously, I need the local GPUs for performance. I already payed for the hardware and I just want to use that.

One of the many problems that arose from my usage of NVIDIA hardware came with [TensorFlow (TF)](https://www.tensorflow.org/). TF is a ML library maintained by Google that can take advantage of GPUs to speed-up the *learning* part. Even laptop GPUs can increase performance from hundreds to thousands of times, as measured on some of my past projects.

So, I got the hardware. There should be some kind of support for TF, right? Yea, kid of. Looking up for it on official sources, you'll be able to find only lengthy and extremely convoluted documentation on the ecosystem. There, what should be a simple install process will turn your life into cross dependency hell.

Basically, in "normal" linux systems, meaning the ones winch use the standard *Linux File System* (LFS), the installation process itself is not extremely complex. However, it has several steps that successively alter the state of you environment at each step. Multiple additions to PATH, environment variables being created, modified and overwritten and other several interactions that modify the current state of a system are the main problem here.

One funny thing that I discovered during several journeys though this sea of shit, is that even the order of dependencies matter. Even if they're not related, do not depend on each other, and are not cited on both sides documentation (TF and Nvidia). They can alter the state of your machine in a way that makes it impossible for TF to recognize your GPU or break steps later on the build/install process. **Yes, it's utter madness.**

![](joker_mad.png)

Finally, I don't want to depend on this hellish process. I can make it work, but I don't want to:
- remember 150 steps to make it work.
- need to write and maintain automation scripts to perform the 150 steps every time.
- be always afraid to update my system fearing the something will break and I'll need to start all over again.
- never know if I'll be able to use my hardware when i rebuild my system, due to version changes on the packages over time.


# Why i use Nix and NixOS

In simple terms: **Nix is very simple, yet extremely powerful.** It's first a package manager with several built-in tools to make your life easier. Furthermore, it's also a programming language that you'll use to interact with the Nix ecosystem.

**You can use Nix for several things, even on distributions other than NixOS.** For instance, use it frequently on my work machine (Ubuntu 22.04) to have isolated and ephemeral environments for several purposes such as: quick setups, test applications, segregated development etc.

**For this reason I've decided to test NixOS.** I'll not go into details here, but the experience has been great, the learning curve was pretty OK, and I just feel that i don't want ever go back.

To put it shortly:
- the feeling of using the nix language is like having the simplicity of bash with great error messages from Rust.
- the feeling of using the nix package manager is like to have a great multi-tool on your pocket. If you're ever in need of pliers, a knife, a Philips screw driver, a toothpick or any other random tool, you're probably covered, and will be able to get the job done.
- the feeling of using NixOS is like having the house of your parents to come back if you're ever is a situation where things in your life have gone terribly wrong. At the same time it's like building your own furniture or fixing-up something on your own house, you'll know exactly what can go wrong, the strong and week spots of it, and how to fix it if something ever go wrong.


# What is the actual problem when running TF with GPUs on NixOS? 

Basically, the documentation is shitty and there is not much support either from Nvidia, the Nix/NixOS community or TensorFlow itself. (I'm looking forward to improve the community side of it)

Basically, what the official documentation from TF and Nvidia suggest is to use a certified docker container to run anything locally. This approach is bad because of several reasons, some of them are:
- It adds overhead reducing performance.
- In adds complexity to an already convoluted environment.
	- If you run TF locally, it has to communicate with CUDA and several other modules which actually manage GPU resources.
	- If you run TF in a container, it has to communicate trough the container, to the internal and external modules, which then manage the GPU resources. In the past I've had already problems with this approach with the container running on distros other than ubuntu.
- It's not reproducible, since TF in the container depends on the Nvidia modules installed on each machine, and the container compatibility with the actual hardware of the machine etc.

So how can we proceed using some kind of Nix/NixOS environment? Yep, I've also asked myself several times and what I've decided to do is:
- Have a simple and stable NixOS system, solely based on a single `config.nix` file, which already have all Nvidia dependencies (drivers, CUDA, TensorRT, ...).
- Use a shell environment based solely on a `shell.nix`, with all the configurations needed to run TensorFlow with Python on any given machine, provided it has the above Nvidia dependencies.


# Solution

The simplest solution would be something like `nix-shell -p python3 -p tensorflow-lite`, which would not work for several reasons. To explain these reasons we'll build a file `shell.nix` which will guide the creation of the shell whenever we call `nix-shell` on terminal.

First, Nvidia and TensorFlow software are not free, so the flag `allowUnfree` should be set. Additionally, the CUDA hardware/software interaction support should be enabled. Hence, the shell should be based on something like:
``` nix
pkgs = import <nixpkgs> {
	config = {
	  allowUnfree = true;
	  cudaSupport = true;
	};
};
```

Furthermore, when using the nix ecosystem with python (or any other sub-ecosystem) you should always look for matched dependency versions. To do so, we construct our `shell.nix` using Python 3.11 from the nix-sore (`pkgs.python311`), and all packages in its bundle `pkgs.python311Packages`, which in this case only regards `pkgs.python311Packages.tensorflow`.

```nix
# shell.nix
let 
  pkgs = import <nixpkgs> {
    config = {
      allowUnfree = true;
      cudaSupport = true;
    };
  };
in
let
  python =    pkgs.python311;
  pp = pkgs.python311Packages;
in
pkgs.mkShell {
  name = "TensorFlow GPU";
  enableParallelBuilding = true;
  
  buildInputs = (
    [ python ]
    ++ (
      with pp; [
        tensorflow
      ]
    )
  );
}
```

## Diagnosis of some problems

The above solution seems to be great, but it had some instabilities for me and this was a huge problem. Between rebuilds of NixOS I was finding my shell not identifying the GPU, or even CUDA not being recognized. With some effort I discovered that sometimes the python versions being used by the shell were not the ones I declared in my `shell.nix` and this was the main root cause of the problem.

There is an environment variable `PYHTONPATH` which usually includes all the installed python versions on the system, including virtual environments and alike. For NixOS this will also include also all the python versions from the nix store that you have available in the current enviroment. So when you call `python something...` on the terminal, you can be calling any of python interpreter from the nix-store that is available.

This, in my case, when running from any terminal within the system without a shell, it looks like:
```bash
[luks@nixos:~]$ echo $PYTHONPATH
# No output.

[luks@nixos:~]$ which python
/run/current-system/sw/bin/python
```

The above output indicates that the python interpreter which should be loaded is in fact what is defined in the current system. Also, it indicates that it is loaded from the path `/run/current-system/sw/bin/python`.

Taking a look at the directory `/run/current-system/sw/bin/`, we can see that the interpreter path points to a symlink to the nix-store as it's expected:
```bash
[luks@nixos:/run/current-system/sw/bin]$ ls -la . | grep "\->" | grep "python -> "
lrwxrwxrwx  3 root root    69 dez 31  1969 python -> /nix/store/nk952l0zags0199a4qysla1k7mqprvi6-python3-3.11.9/bin/python
```

Therefore, any python script I run in my system simply calling `python something...` will use the interpreter contained in `nk952l0zags0199a4qysla1k7mqprvi6-python3-3.11.9`.

**An important thing should be pointed now. As shown before, there is no connection to any other python interpreter when running anything on my overall system.** From now on, I'll be running these diagnostics on my nix-shell created above.

```nix
[luks@nixos:~]$ nix-shell
trace: warning: cudaPackages.autoAddDriverRunpath is deprecated, use pkgs.autoAddDriverRunpath instead

[nix-shell:~]$ echo $PYTHONPATH
/nix/store/qg6f0lxm5l2x6mm4gghjwgi15hal8rry-python3.11-cffi-1.16.0/lib/python3.11/site-packages:/nix/store/77qfs74qz4x077j6ssv2xyf6k9ca0aik-python3.11-pycparser-2.22/lib/python3.11/site-packages:/nix/store/5w07wfs288qpmnvjywk24f3ak5k1np7r-python3-3.11.9/lib/python3.11/site-packages:/nix/store/35vj42sjcz6d78z88k9q5hifxj34m4qq-python3.11-tensorflow-gpu-2.13.0/lib/python3.11/site-packages:/nix/store/6km93zw8kr34jlay4v0d8594fppsmpm4-python3.11-absl-py-2.1.0/lib/python3.11/site-packages:/nix/store/4rwyk9fdsnapwx399jzcdm16f7n6kirx-python3.11-six-1.16.0/lib/python3.11/site-packages:/nix/store/9qa0803z75v7cys619fky0s3knczc3sd-python3.11-astunparse-1.6.3/lib/python3.11/site-packages:/nix/store/xhabz028lqc8jajqa4p9s5wr5swbwxc8-python3.11-wheel-0.43.0/lib/python3.11/site-packages:/nix/store/78mr2jjwcgnmaklzb19p4d0igbv3vjq7-python3.11-flatbuffers-23.5.26/lib/python3.11/site-packages:/nix/store/fg7xpvqwnfgcii1iw9q0griglwmncksy-python3.11-gast-0.5.3/lib/python3.11/site-packages:/nix/store/705lxxbspndym9c1b83zl4whpbp6k9qh-python3.11-google-pasta-0.2.0/lib/python3.11/site-packages:/nix/store/ybnprsmxlp77nqgsf0g37b6m6dvz4g7c-python3.11-grpcio-1.62.2/lib/python3.11/site-packages:/nix/store/zrfz4rpsrqajggdzcf1vry460jgys5h3-python3.11-h5py-3.11.0/lib/python3.11/site-packages:/nix/store/0kk9b2fk9qa9w1nylwq066n8qj20vyqr-python3.11-numpy-1.26.4/lib/python3.11/site-packages:/nix/store/9j25lambqqg0q324c8jmkllhfihdpsiw-python3.11-keras-preprocessing-1.1.2/lib/python3.11/site-packages:/nix/store/9vvax258hzm8q1c3ps34vkl3fyaxqsci-python3.11-scipy-1.13.0/lib/python3.11/site-packages:/nix/store/dhasy2sg8p1z0nmpriydqvw1yg4bb0dl-python3.11-pillow-10.3.0/lib/python3.11/site-packages:/nix/store/9z4pvshljvpfdc8g1bmvm2pqk8i43hd0-python3.11-olefile-0.47/lib/python3.11/site-packages:/nix/store/43vn99zb01jvbzbslf039qgh8wi35p2m-python3.11-defusedxml-0.7.1/lib/python3.11/site-packages:/nix/store/fxbpw2qdsszsfs7zsw1vq0z80jmj5vvy-python3.11-opt-einsum-3.3.0/lib/python3.11/site-packages:/nix/store/g5c193194xl1p0w5i3rjmsirxb5r4dj4-python3.11-packaging-24.0/lib/python3.11/site-packages:/nix/store/xnvy6l68pqliaw20ymxivmf4d46di452-python3.11-protobuf-4.21.12/lib/python3.11/site-packages:/nix/store/58q5i3dhni3glp617gx82q8fymb5gn9q-python3.11-tensorflow-estimator-2.11.0/lib/python3.11/site-packages:/nix/store/pmxfwdp962z86w4g87vb44v56yhlbr2l-python3.11-mock-5.1.0/lib/python3.11/site-packages:/nix/store/sgr7z9ljcvrsgmvzjkwl90ayxx1kqp8b-python3.11-termcolor-2.4.0/lib/python3.11/site-packages:/nix/store/wsx1ylbvdbzxkqw7mbwsky6x8zfi1nny-python3.11-typing-extensions-4.11.0/lib/python3.11/site-packages:/nix/store/i26hmnv49r865jsghaa3g93z808vij4k-python3.11-wrapt-1.16.0/lib/python3.11/site-packages:/nix/store/2xvznvvi77jp72060rfmgjnafmb4q2lz-python3.11-tensorboard-2.16.2/lib/python3.11/site-packages:/nix/store/bk26k8s3ld6il2261f3xk7m63s7cy64g-python3.11-google-auth-oauthlib-1.2.0/lib/python3.11/site-packages:/nix/store/7n5izx4qw24xnjlr5bd6dpjalk7x9yj4-python3.11-google-auth-2.29.0/lib/python3.11/site-packages:/nix/store/gsvnnabzlzn1xfrqvby9h84qqywiagb7-python3.11-cachetools-5.3.2/lib/python3.11/site-packages:/nix/store/3q85xwwc0fq5p1r4sfaawb84cn2zwgrb-python3.11-pyasn1-modules-0.4.0/lib/python3.11/site-packages:/nix/store/sc2ni7i2z4dmkackqbhcw9vjk3lbzcxs-python3.11-pyasn1-0.6.0/lib/python3.11/site-packages:/nix/store/2g9skvsly3625fv3bxw8yfdpw31wapkc-python3.11-rsa-4.9/lib/python3.11/site-packages:/nix/store/niv7v4474p7krdx1bnf6jg3wkxcq913m-python3.11-requests-oauthlib-1.3.1/lib/python3.11/site-packages:/nix/store/2s3zx4gff5gwsjqi1rs5ryjsgnimd6mk-python3.11-oauthlib-3.2.2/lib/python3.11/site-packages:/nix/store/gzq896zaivy310ww92xzrbn9m5s26s0w-python3.11-requests-2.31.0/lib/python3.11/site-packages:/nix/store/yi8vzx7y5r2hmk73lmbwp7yc4bmmrnjk-python3.11-brotlicffi-1.1.0.0/lib/python3.11/site-packages:/nix/store/03ifp6dsviybcg80krp71x0dk7j62h6s-python3.11-certifi-2024.02.02/lib/python3.11/site-packages:/nix/store/0vdrvslhblawvgj4xqgpzq3fylyhpci8-python3.11-charset-normalizer-3.3.2/lib/python3.11/site-packages:/nix/store/cwy1h3da89id2b6i1frv1ggb9yyyd2np-python3.11-idna-3.7/lib/python3.11/site-packages:/nix/store/yp6bx6fj0ix9z3pna04pfjmhfcyx5sxb-python3.11-urllib3-2.2.1/lib/python3.11/site-packages:/nix/store/lkkk5qvdz5vky78w3i0vbrndfqjrgffs-python3.11-markdown-3.6/lib/python3.11/site-packages:/nix/store/2amskndx53q9vwgy4h4snjhqng2ckdx8-python3.11-setuptools-69.5.1/lib/python3.11/site-packages:/nix/store/5207kxyhgzx3x815dvhippz0wjwh6mch-python3.11-tensorboard-data-server-0.7.2/lib/python3.11/site-packages:/nix/store/c0wf639jjjmhyv76kc50sjxdb0nwp4d5-python3.11-tensorboard_plugin_profile-2.11.1/lib/python3.11/site-packages:/nix/store/nm7vi8f4aalg5y2yn5h312s48f2s5xjq-python3.11-gviz_api-1.10.0/lib/python3.11/site-packages:/nix/store/pkhnsm4pg6w5cf3b8d4324fb8xsm20hr-python3.11-werkzeug-3.0.3/lib/python3.11/site-packages:/nix/store/l1ambfmvdj9n4fnxn86vlx8xkb612y5g-python3.11-markupsafe-2.1.5/lib/python3.11/site-packages:/nix/store/p97ns4agwhl78x1nk7ir1b9j3f6w5mc1-python3.11-tensorboard_plugin_wit-1.7.0/lib/python3.11/site-packages
```

Contrary from what is expected, the output above shows paths for **a lot** of python packages in my system. This should not be the case since the shell already has a python package specified on `shell.nix`. Additionally, the python interpreter contained as standard in the shell is `5w07wfs288qpmnvjywk24f3ak5k1np7r-python3-3.11.9`:
```bash
[nix-shell:~]$ which python
/nix/store/5w07wfs288qpmnvjywk24f3ak5k1np7r-python3-3.11.9/bin/python
```

So, finally, what seems to be the problem? I'm not completely sure yet, but a simple solution I found is to make sure the python interpreter path is contained on the environment variable `PYTHONPATH` inside the shell. In doing so, the correct python version is loaded whenever a script calls it.

It seems to be the case that the libraries involved in the Nvidia + TensorFlow gigantic castle of cards sometimes use `PYTHONPATH`. If this environment variable is not set correctly, they cannot communicate with te interpreter version running inside the shell. (by the way, fuck u Nvidia)


## Shell hooks

A simple way to automate this procedure of setting `PYTHONPATH` is to use shell hooks. There are mainly two types of shell hooks to be used with nix shells, they are `shellHook` which is used when the shell is activated, and `postShellHook` which is used right after the shell setup has been completed.

```nix
  postShellHook = ''
    export PYTHONPATH=$(which python)
  '';

```



preShellHook: Hook to execute commands before shellHook.
postShellHook: Hook to execute commands after shellHook.


# 