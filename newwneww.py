[     UTC     ] Logs for chatbot-gaib-c2zeu8h8apyjysgwfms6jn.streamlit.app/

────────────────────────────────────────────────────────────────────────────────────────

[10:30:31] 🖥 Provisioning machine...

[10:30:31] 🎛 Preparing system...

[10:30:31] ⛓ Spinning up manager process...

[10:30:32] 🚀 Starting up repository: 'chatbot-gaib', branch: 'main', main module: 'newwneww.py'

[10:30:32] 🐙 Cloning repository...

[10:30:33] 🐙 Cloning into '/mount/src/chatbot-gaib'...

[10:30:33] 🐙 Cloned repository!

[10:30:33] 🐙 Pulling code changes from Github...

[10:30:34] 📦 Processing dependencies...


──────────────────────────────────────── uv ───────────────────────────────────────────


Using uv pip install.

Using Python 3.13.5 environment at /home/adminuser/venv

Resolved [2025-08-20 10:30:38.486529] 287 packages[2025-08-20 10:30:38.486837]  [2025-08-20 10:30:38.487147] in 3.75s[2025-08-20 10:30:38.487476] 

  × Failed to download and build `pywinpty==2.0.14`

  ╰─▶ Build backend failed to build wheel through `build_wheel` (exit status:

      1)


      [stdout]

      Running `maturin pep517 build-wheel -i

      /home/adminuser/.cache/uv/builds-v0/.tmpP3IsCQ/bin/python

      --compatibility off`

      Rust not found, installing into a temporary directory

      cargo 1.89.0 (c24e10642 2025-06-23)


      [stderr]

      Python reports SOABI: cpython-313-x86_64-linux-gnu

      Computed rustc target triple: x86_64-unknown-linux-gnu

      Installation directory: /home/adminuser/.cache/puccinialin

      Downloading rustup-init from

      https://static.rust-lang.org/rustup/dist/x86_64-unknown-linux-gnu/rustup-init

Downloading rustup-init:   0%|          | 0.00/20.9M [00:00<?,

Downloading rustup-init:  44%|████▍     | 9.15M/20.9M [00:00<00:00,

Downloading rustup-init: 100%|██████████| 20.9M/20.9M

      [00:00<00:00, 109MB/s]

      Installing rust to /home/adminuser/.cache/puccinialin/rustup

      info: profile set to 'minimal'

      info: default host triple is x86_64-unknown-linux-gnu

      info: syncing channel updates for 'stable-x86_64-unknown-linux-gnu'

      info: latest update on 2025-08-07, rust version 1.89.0 (29483883e

      2025-08-04)

      info: downloading component 'cargo'

      info: downloading component 'rust-std'

      info: downloading component 'rustc'

      info: installing component 'cargo'

      info: installing component 'rust-std'

      info: installing component 'rustc'

      info: default toolchain set to 'stable-x86_64-unknown-linux-gnu'

      Checking if cargo is installed

          Updating crates.io index

       Downloading crates ...

        Downloaded enum-primitive-derive v0.3.0

        Downloaded errno v0.3.9

        Downloaded either v1.13.0

        Downloaded bitflags v2.6.0

        Downloaded winpty-rs v0.4.0

        Downloaded windows-implement v0.58.0

        Downloaded windows-strings v0.1.0

        Downloaded windows-core v0.58.0

        Downloaded unicode-ident v1.0.13

        Downloaded pyo3-macros-backend v0.22.5

        Downloaded pyo3-ffi v0.22.5

        Downloaded proc-macro2 v1.0.88

        Downloaded[2025-08-20 10:31:22.938654]  once_cell v1.20.2

        Downloaded which v6.0.3

        Downloaded portable-atomic v1.9.0

        Downloaded target-lexicon v0.12.16

        Downloaded quote v1.0.37

        Downloaded pyo3-build-config v0.22.5

        Downloaded num-traits v0.2.19

        Downloaded memoffset v0.9.1

        Downloaded syn v2.0.79

        Downloaded windows-result v0.2.0

        Downloaded windows-interface v0.58.0

        Downloaded home v0.5.9

        Downloaded heck v0.5.0

        Downloaded rustix v0.38.37

        Downloaded windows-targets v0.52.6

        Downloaded windows_x86_64_gnullvm v0.52.6

        Downloaded windows_i686_gnullvm v0.52.6

        Downloaded windows_aarch64_gnullvm v0.52.6

        Downloaded unindent v0.2.3

        Downloaded winsafe v0.0.19

        Downloaded pyo3-macros v0.22.5

        Downloaded indoc v2.0.5

        Downloaded autocfg v1.4.0

        Downloaded pyo3 v0.22.5

        Downloaded cfg-if v1.0.0

        Downloaded libc v0.2.160

        Downloaded windows_x86_64_msvc v0.52.6

        Downloaded windows_x86_64_gnu v0.52.6

        Downloaded windows_i686_gnu v0.52.6

        Downloaded windows_aarch64_msvc v0.52.6

        Downloaded windows_i686_msvc v0.52.6

        Downloaded linux-raw-sys v0.4.14

        Downloaded windows-sys v0.52.0

        Downloaded windows v0.58.0

      ⚠️  Warning: `project.version` field is required in pyproject.toml unless

      it is present in the `project.dynamic` list

      📦 Including license file `LICENSE.txt`

      🍹 Building a mixed python/rust project

      🔗 Found pyo3 bindings

      🐍 Found CPython 3.13 at

      /home/adminuser/.cache/uv/builds-v0/.tmpP3IsCQ/bin/python

         Compiling proc-macro2 v1.0.88

         Compiling windows_x86_64_gnu v0.52.6

         Compiling unicode-ident v1.0.13

         Compiling target-lexicon v0.12.16

         Compiling autocfg v1.4.0

         Compiling once_cell v1.20.2

         Compiling rustix v0.38.37

         Compiling linux-raw-sys v0.4.14

         Compiling bitflags v2.6.0

         Compiling libc v0.2.160

         Compiling home v0.5.9

         Compiling either v1.13.0

         Compiling heck v0.5.0

         Compiling indoc v2.0.5

         Compiling unindent v0.2.3

         Compiling cfg-if v1.0.0

         Compiling num-traits v0.2.19

         Compiling[2025-08-20 10:31:22.939006]  memoffset v0.9.1

         Compiling windows-targets v0.52.6

         Compiling windows-result v0.2.0

         Compiling windows-strings v0.1.0

         Compiling pyo3-build-config v0.22.5

         Compiling quote v1.0.37

         Compiling syn v2.0.79

         Compiling pyo3-macros-backend v0.22.5

         Compiling pyo3-ffi v0.22.5

         Compiling pyo3 v0.22.5

         Compiling which v6.0.3

         Compiling windows-interface v0.58.0

         Compiling windows-implement v0.58.0

         Compiling enum-primitive-derive v0.3.0

         Compiling windows-core v0.58.0

         Compiling windows v0.58.0

         Compiling winpty-rs v0.4.0

      error[E0433]: failed to resolve: could not find `Win32` in `windows`

       -->

      /home/adminuser/.cache/puccinialin/cargo/registry/src/index.crates.io-1949cf8c6b5b557f/winpty-rs-0.4.0/src/pty/base.rs:3:14

        |

      3 | use windows::Win32::Foundation::{CloseHandle, HANDLE,

      STATUS_PENDING, S_OK, WAIT_FAILED, WAIT_OBJECT_0, WAIT_TIMEOUT};

        |              ^^^^^ could not find `Win32` in `windows`


      error[E0433]: failed to resolve: could not find `Win32` in `windows`

       -->

      /home/adminuser/.cache/puccinialin/cargo/registry/src/index.crates.io-1949cf8c6b5b557f/winpty-rs-0.4.0/src/pty/base.rs:4:14

        |

      4 | use windows::Win32::Storage::FileSystem::{GetFileSizeEx, ReadFile,

      WriteFile};

        |              ^^^^^ could not find `Win32` in `windows`


      error[E0433]: failed to resolve: could not find `Win32` in `windows`

       -->

      /home/adminuser/.cache/puccinialin/cargo/registry/src/index.crates.io-1949cf8c6b5b557f/winpty-rs-0.4.0/src/pty/base.rs:5:14

        |

      5 | use windows::Win32::System::Pipes::PeekNamedPipe;

        |[2025-08-20 10:31:22.939232]               ^^^^^ could not find `Win32` in `windows`


      error[E0433]: failed to resolve: could not find `Win32` in `windows`

       -->

      /home/adminuser/.cache/puccinialin/cargo/registry/src/index.crates.io-1949cf8c6b5b557f/winpty-rs-0.4.0/src/pty/base.rs:6:14

        |

      6 | use windows::Win32::System::IO::CancelIoEx;

        |              ^^^^^ could not find `Win32` in `windows`


      error[E0433]: failed to resolve: could not find `Win32` in `windows`

       -->

      /home/adminuser/.cache/puccinialin/cargo/registry/src/index.crates.io-1949cf8c6b5b557f/winpty-rs-0.4.0/src/pty/base.rs:7:14

        |

      7 | use windows::Win32::System::Threading::{GetExitCodeProcess,

      GetProcessId, WaitForSingleObject};

        |              ^^^^^ could not find `Win32` in `windows`


      error[E0433]: failed to resolve: could not find `Win32` in `windows`

       -->

      /home/adminuser/.cache/puccinialin/cargo/registry/src/index.crates.io-1949cf8c6b5b557f/winpty-rs-0.4.0/src/pty/base.rs:8:14

        |

      8 |[2025-08-20 10:31:22.939447]  use windows::Win32::Globalization::{MultiByteToWideChar,

      WideCharToMultiByte, CP_UTF8, MULTI_BYTE_TO_WIDE_CHAR_FLAGS};

        |              ^^^^^ could not find `Win32` in `windows`


      error[E0433]: failed to resolve: could not find `Win32` in `windows`

        -->

      /home/adminuser/.cache/puccinialin/cargo/registry/src/index.crates.io-1949cf8c6b5b557f/winpty-rs-0.4.0/src/pty/base.rs:10:14

         |

      10 | use windows::Win32::System::Threading::INFINITE;

         |              ^^^^^ could not find `Win32` in `windows`


      error[E0432]: unresolved import `windows::core`

       -->

      /home/adminuser/.cache/puccinialin/cargo/registry/src/index.crates.io-1949cf8c6b5b557f/winpty-rs-0.4.0/src/pty/base.rs:9:14

        |

      9 | use windows::core::{HRESULT, Error, PCSTR};

        |              ^^^^ could not find `core` in `windows`


      error: the `Self` constructor can only be used with tuple or unit

      structs

        -->

      /home/adminuser/.cache/puccinialin/cargo/registry/src/index.crates.io-1949cf8c6b5b557f/winpty-rs-0.4.0/src/pty/base.rs:70:9

         |

      70 |         Self(value.0)

         |         ^^^^^^^^^^^^^


      Some errors have detailed explanations: E0432, E0433.

      For more information about an error, try `rustc --explain E0432`.

      error: could not compile `winpty-rs` (lib) due to 9 previous errors

      warning[2025-08-20 10:31:22.939675] : build failed, waiting for other jobs to finish...

      💥 maturin failed

        Caused by: Failed to build a native library through cargo

        Caused by: Cargo build finished with "exit status: 101":

      `env -u CARGO PYO3_ENVIRONMENT_SIGNATURE="cpython-3.13-64bit"

      PYO3_PYTHON="/home/adminuser/.cache/uv/builds-v0/.tmpP3IsCQ/bin/python"

      PYTHON_SYS_EXECUTABLE="/home/adminuser/.cache/uv/builds-v0/.tmpP3IsCQ/bin/python"

      "cargo" "rustc" "--message-format"

      "json-render-diagnostics" "--manifest-path"

      "/home/adminuser/.cache/uv/sdists-v6/pypi/pywinpty/2.0.14/kH1wVMrpfysfldyou9NvN/src/Cargo.toml"

      "--release" "--lib"`

      Error: command ['maturin', 'pep517', 'build-wheel', '-i',

      '/home/adminuser/.cache/uv/builds-v0/.tmpP3IsCQ/bin/python',

      '--compatibility', 'off'] returned non-zero exit status 1


Checking if Streamlit is installed

Installing rich for an improved exception logging

Using uv pip install.

Using Python 3.13.5 environment at /home/adminuser/venv

Resolved 4 packages in 62ms

Prepared 3 packages in 114ms

Installed 4 packages in 17ms

 + markdown-it-py==4.0.0

 + mdurl==0.1.2

 + pygments==2.19.2

 + rich==14.1.0


────────────────────────────────────────────────────────────────────────────────────────



──────────────────────────────────────── pip ───────────────────────────────────────────


Using standard pip install.

Collecting aiohappyeyeballs==2.6.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 1))

  Downloading aiohappyeyeballs-2.6.1-py3-none-any.whl.metadata (5.9 kB)

Collecting aiohttp==3.11.16 (from -r /mount/src/chatbot-gaib/requirements.txt (line 2))

  Downloading aiohttp-3.11.16-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (7.7 kB)

Collecting aiosignal==1.3.2 (from -r /mount/src/chatbot-gaib/requirements.txt (line 3))

  Downloading aiosignal-1.3.2-py2.py3-none-any.whl.metadata (3.8 kB)

Collecting altair==5.5.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 4))

  Downloading altair-5.5.0-py3-none-any.whl.metadata (11 kB)

Collecting annotated-types==0.7.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 5))

  Downloading annotated_types-0.7.0-py3-none-any.whl.metadata (15 kB)

Collecting anyio==4.8.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 6))

  Downloading anyio-4.8.0-py3-none-any.whl.metadata (4.6 kB)

Collecting argon2-cffi==23.1.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 7))

  Downloading argon2_cffi-23.1.0-py3-none-any.whl.metadata (5.2 kB)

Collecting argon2-cffi-bindings==21.2.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 8))

  Downloading argon2_cffi_bindings-21.2.0-cp36-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.7 kB)

Collecting arrow==1.3.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 9))

  Downloading arrow-1.3.0-py3-none-any.whl.metadata (7.5 kB)

Collecting astroid==3.3.10 (from -r /mount/src/chatbot-gaib/requirements.txt (line 10))

  Downloading astroid-3.3.10-py3-none-any.whl.metadata (4.4 kB)

Collecting asttokens==3.0.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 11))

  Downloading asttokens-3.0.0-py3-none-any.whl.metadata (4.7 kB)

Collecting async-lru==2.0.4 (from -r /mount/src/chatbot-gaib/requirements.txt (line 12))

  Downloading async_lru-2.0.4-py3-none-any.whl.metadata (4.5 kB)

Collecting attrs==24.3.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 13))

  Downloading attrs-24.3.0-py3-none-any.whl.metadata (11 kB)

Collecting babel==2.16.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 14))

  Downloading babel-2.16.0-py3-none-any.whl.metadata (1.5 kB)

Collecting backoff==2.2.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 15))

  Downloading backoff-2.2.1-py3-none-any.whl.metadata (14 kB)

Collecting bcrypt==4.3.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 16))

  Downloading bcrypt-4.3.0-cp39-abi3-manylinux_2_28_x86_64.whl.metadata (10 kB)

Collecting beautifulsoup4==4.12.3 (from -r /mount/src/chatbot-gaib/requirements.txt (line 17))

  Downloading beautifulsoup4-4.12.3-py3-none-any.whl.metadata (3.8 kB)

Collecting black==25.1.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 18))

  Downloading black-25.1.0-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.manylinux_2_28_x86_64.whl.metadata (81 kB)

     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 81.3/81.3 kB 11.1 MB/s eta 0:00:00[2025-08-20 10:31:27.577261] 

Collecting bleach==6.2.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 19))

  Downloading bleach-6.2.0-py3-none-any.whl.metadata (30 kB)

Collecting blinker==1.9.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 20))

  Downloading blinker-1.9.0-py3-none-any.whl.metadata (1.6 kB)

Collecting Bottleneck==1.5.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 21))

  Downloading bottleneck-1.5.0-cp313-cp313-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (8.1 kB)

Collecting build==1.2.2.post1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 22))

  Downloading build-1.2.2.post1-py3-none-any.whl.metadata (6.5 kB)

Collecting cachetools==5.5.2 (from -r /mount/src/chatbot-gaib/requirements.txt (line 23))

  Downloading cachetools-5.5.2-py3-none-any.whl.metadata (5.4 kB)

Collecting certifi==2025.6.15 (from -r /mount/src/chatbot-gaib/requirements.txt (line 24))

  Downloading certifi-2025.6.15-py3-none-any.whl.metadata (2.4 kB)

Collecting cffi==1.17.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 25))

  Downloading cffi-1.17.1-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (1.5 kB)

Collecting charset-normalizer==3.4.2 (from -r /mount/src/chatbot-gaib/requirements.txt (line 26))

  Downloading charset_normalizer-3.4.2-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (35 kB)

Collecting chromadb==1.0.15 (from -r /mount/src/chatbot-gaib/requirements.txt (line 27))

  Downloading chromadb-1.0.15-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (7.0 kB)

Collecting click==8.1.8 (from -r /mount/src/chatbot-gaib/requirements.txt (line 28))

  Downloading click-8.1.8-py3-none-any.whl.metadata (2.3 kB)

Collecting colorama==0.4.6 (from -r /mount/src/chatbot-gaib/requirements.txt (line 29))

  Downloading colorama-0.4.6-py2.py3-none-any.whl.metadata (17 kB)

Collecting coloredlogs==15.0.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 30))

  Downloading coloredlogs-15.0.1-py2.py3-none-any.whl.metadata (12 kB)

Collecting comm==0.2.2 (from -r /mount/src/chatbot-gaib/requirements.txt (line 31))

  Downloading comm-0.2.2-py3-none-any.whl.metadata (3.7 kB)

Collecting contourpy==1.3.2 (from -r /mount/src/chatbot-gaib/requirements.txt (line 32))

  Downloading contourpy-1.3.2-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (5.5 kB)

Collecting cryptography==44.0.2 (from -r /mount/src/chatbot-gaib/requirements.txt (line 33))

  Downloading cryptography-44.0.2-cp39-abi3-manylinux_2_28_x86_64.whl.metadata (5.7 kB)

Collecting cycler==0.12.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 34))

  Downloading cycler-0.12.1-py3-none-any.whl.metadata (3.8 kB)

Collecting dataclasses-json==0.6.7 (from -r /mount/src/chatbot-gaib/requirements.txt (line 35))

  Downloading dataclasses_json-0.6.7-py3-none-any.whl.metadata (25 kB)

Collecting db-dtypes==1.4.3 (from -r /mount/src/chatbot-gaib/requirements.txt (line 36))

  Downloading db_dtypes-1.4.3-py3-none-any.whl.metadata (3.0 kB)

Collecting debugpy==1.8.12 (from -r /mount/src/chatbot-gaib/requirements.txt (line 37))

  Downloading debugpy-1.8.12-cp313-cp313-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (1.3 kB)

Collecting decorator==5.1.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 38))

  Downloading decorator-5.1.1-py3-none-any.whl.metadata (4.0 kB)

Collecting defusedxml==0.7.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 39))

  Downloading defusedxml-0.7.1-py2.py3-none-any.whl.metadata (32 kB)

Collecting dill==0.4.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 40))

  Downloading dill-0.4.0-py3-none-any.whl.metadata (10 kB)

Collecting distro==1.9.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 41))

  Downloading distro-1.9.0-py3-none-any.whl.metadata (6.8 kB)

Collecting docstring_parser==0.16 (from -r /mount/src/chatbot-gaib/requirements.txt (line 42))

  Downloading docstring_parser-0.16-py3-none-any.whl.metadata (3.0 kB)

Collecting durationpy==0.10 (from -r /mount/src/chatbot-gaib/requirements.txt (line 43))

  Downloading durationpy-0.10-py3-none-any.whl.metadata (340 bytes)

Collecting et_xmlfile==2.0.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 44))

  Downloading et_xmlfile-2.0.0-py3-none-any.whl.metadata (2.7 kB)

Collecting executing==2.2.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 45))

  Downloading executing-2.2.0-py2.py3-none-any.whl.metadata (8.9 kB)

Collecting Faker==35.0.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 46))

  Downloading Faker-35.0.0-py3-none-any.whl.metadata (15 kB)

Collecting fastjsonschema==2.21.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 47))

  Downloading fastjsonschema-2.21.1-py3-none-any.whl.metadata (2.2 kB)

Collecting filelock==3.18.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 48))

  Downloading filelock-3.18.0-py3-none-any.whl.metadata (2.9 kB)

Collecting filetype==1.2.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 49))

  Downloading filetype-1.2.0-py2.py3-none-any.whl.metadata (6.5 kB)

Collecting Flask==3.1.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 50))

  Downloading flask-3.1.0-py3-none-any.whl.metadata (2.7 kB)

Collecting flatbuffers==25.2.10 (from -r /mount/src/chatbot-gaib/requirements.txt (line 51))

  Downloading flatbuffers-25.2.10-py2.py3-none-any.whl.metadata (875 bytes)

Collecting fonttools==4.59.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 52))

  Downloading fonttools-4.59.0-cp313-cp313-manylinux1_x86_64.manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_5_x86_64.whl.metadata (107 kB)

     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 107.9/107.9 kB 117.0 MB/s eta 0:00:00[2025-08-20 10:31:30.698210] 

Collecting fqdn==1.5.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 53))

  Downloading fqdn-1.5.1-py3-none-any.whl.metadata (1.4 kB)

Collecting frozenlist==1.5.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 54))

  Downloading frozenlist-1.5.0-cp313-cp313-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (13 kB)

Collecting fsspec==2025.7.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 55))

  Downloading fsspec-2025.7.0-py3-none-any.whl.metadata (12 kB)

Collecting gitdb==4.0.12 (from -r /mount/src/chatbot-gaib/requirements.txt (line 56))

  Downloading gitdb-4.0.12-py3-none-any.whl.metadata (1.2 kB)

Collecting GitPython==3.1.44 (from -r /mount/src/chatbot-gaib/requirements.txt (line 57))

  Downloading GitPython-3.1.44-py3-none-any.whl.metadata (13 kB)

Collecting google-ai-generativelanguage==0.6.18 (from -r /mount/src/chatbot-gaib/requirements.txt (line 58))

  Downloading google_ai_generativelanguage-0.6.18-py3-none-any.whl.metadata (9.8 kB)

Collecting google-api-core==2.25.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 59))

  Downloading google_api_core-2.25.1-py3-none-any.whl.metadata (3.0 kB)

Collecting google-auth==2.40.3 (from -r /mount/src/chatbot-gaib/requirements.txt (line 60))

  Downloading google_auth-2.40.3-py2.py3-none-any.whl.metadata (6.2 kB)

Collecting google-auth-oauthlib==1.2.2 (from -r /mount/src/chatbot-gaib/requirements.txt (line 61))

  Downloading google_auth_oauthlib-1.2.2-py3-none-any.whl.metadata (2.7 kB)

Collecting google-cloud-aiplatform==1.105.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 62))

  Downloading google_cloud_aiplatform-1.105.0-py2.py3-none-any.whl.metadata (38 kB)

Collecting google-cloud-bigquery==3.34.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 63))

  Downloading google_cloud_bigquery-3.34.0-py3-none-any.whl.metadata (8.0 kB)

Collecting google-cloud-bigquery-storage==2.32.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 64))

  Downloading google_cloud_bigquery_storage-2.32.0-py3-none-any.whl.metadata (9.8 kB)

Collecting google-cloud-core==2.4.3 (from -r /mount/src/chatbot-gaib/requirements.txt (line 65))

  Downloading google_cloud_core-2.4.3-py2.py3-none-any.whl.metadata (2.7 kB)

Collecting google-cloud-resource-manager==1.14.2 (from -r /mount/src/chatbot-gaib/requirements.txt (line 66))

  Downloading google_cloud_resource_manager-1.14.2-py3-none-any.whl.metadata (9.6 kB)

Collecting google-cloud-storage==2.19.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 67))

  Downloading google_cloud_storage-2.19.0-py2.py3-none-any.whl.metadata (9.1 kB)

Collecting google-crc32c==1.7.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 68))

  Downloading google_crc32c-1.7.1-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (2.3 kB)

Collecting google-genai==1.21.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 69))

  Downloading google_genai-1.21.1-py3-none-any.whl.metadata (37 kB)

Collecting google-resumable-media==2.7.2 (from -r /mount/src/chatbot-gaib/requirements.txt (line 70))

  Downloading google_resumable_media-2.7.2-py2.py3-none-any.whl.metadata (2.2 kB)

Collecting googleapis-common-protos==1.70.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 71))

  Downloading googleapis_common_protos-1.70.0-py3-none-any.whl.metadata (9.3 kB)

Collecting greenlet==3.2.3 (from -r /mount/src/chatbot-gaib/requirements.txt (line 72))

  Downloading greenlet-3.2.3-cp313-cp313-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl.metadata (4.1 kB)

Collecting groq==0.28.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 73))

  Downloading groq-0.28.0-py3-none-any.whl.metadata (15 kB)

Collecting grpc-google-iam-v1==0.14.2 (from -r /mount/src/chatbot-gaib/requirements.txt (line 74))

  Downloading grpc_google_iam_v1-0.14.2-py3-none-any.whl.metadata (9.1 kB)

Collecting grpcio==1.73.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 75))

  Downloading grpcio-1.73.0-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (3.8 kB)

Collecting grpcio-status==1.73.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 76))

  Downloading grpcio_status-1.73.0-py3-none-any.whl.metadata (1.1 kB)

Collecting h11==0.14.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 77))

  Downloading h11-0.14.0-py3-none-any.whl.metadata (8.2 kB)

Collecting httpcore==1.0.7 (from -r /mount/src/chatbot-gaib/requirements.txt (line 78))

  Downloading httpcore-1.0.7-py3-none-any.whl.metadata (21 kB)

Collecting httptools==0.6.4 (from -r /mount/src/chatbot-gaib/requirements.txt (line 79))

  Downloading httptools-0.6.4-cp313-cp313-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (3.6 kB)

Collecting httpx==0.28.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 80))

  Downloading httpx-0.28.1-py3-none-any.whl.metadata (7.1 kB)

Collecting httpx-sse==0.4.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 81))

  Downloading httpx_sse-0.4.0-py3-none-any.whl.metadata (9.0 kB)

Collecting huggingface-hub==0.33.4 (from -r /mount/src/chatbot-gaib/requirements.txt (line 82))

  Downloading huggingface_hub-0.33.4-py3-none-any.whl.metadata (14 kB)

Collecting humanfriendly==10.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 83))

  Downloading humanfriendly-10.0-py2.py3-none-any.whl.metadata (9.2 kB)

Collecting idna==3.10 (from -r /mount/src/chatbot-gaib/requirements.txt (line 84))

  Downloading idna-3.10-py3-none-any.whl.metadata (10 kB)

Collecting importlib_metadata==8.7.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 85))

  Downloading importlib_metadata-8.7.0-py3-none-any.whl.metadata (4.8 kB)

Collecting importlib_resources==6.5.2 (from -r /mount/src/chatbot-gaib/requirements.txt (line 86))

  Downloading importlib_resources-6.5.2-py3-none-any.whl.metadata (3.9 kB)

Collecting ipykernel==6.29.5 (from -r /mount/src/chatbot-gaib/requirements.txt (line 87))

  Downloading ipykernel-6.29.5-py3-none-any.whl.metadata (6.3 kB)

Collecting ipython==8.31.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 88))

  Downloading ipython-8.31.0-py3-none-any.whl.metadata (4.9 kB)

Collecting ipywidgets==8.1.5 (from -r /mount/src/chatbot-gaib/requirements.txt (line 89))

  Downloading ipywidgets-8.1.5-py3-none-any.whl.metadata (2.3 kB)

Collecting isoduration==20.11.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 90))

  Downloading isoduration-20.11.0-py3-none-any.whl.metadata (5.7 kB)

Collecting isort==6.0.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 91))

  Downloading isort-6.0.1-py3-none-any.whl.metadata (11 kB)

Collecting itsdangerous==2.2.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 92))

  Downloading itsdangerous-2.2.0-py3-none-any.whl.metadata (1.9 kB)

Collecting jedi==0.19.2 (from -r /mount/src/chatbot-gaib/requirements.txt (line 93))

  Downloading jedi-0.19.2-py2.py3-none-any.whl.metadata (22 kB)

Collecting Jinja2==3.1.5 (from -r /mount/src/chatbot-gaib/requirements.txt (line 94))

  Downloading jinja2-3.1.5-py3-none-any.whl.metadata (2.6 kB)

Collecting jiter==0.9.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 95))

  Downloading jiter-0.9.0-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (5.2 kB)

Collecting joblib==1.5.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 96))

  Downloading joblib-1.5.1-py3-none-any.whl.metadata (5.6 kB)

Collecting json5==0.10.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 97))

  Downloading json5-0.10.0-py3-none-any.whl.metadata (34 kB)

Collecting jsonpatch==1.33 (from -r /mount/src/chatbot-gaib/requirements.txt (line 98))

  Downloading jsonpatch-1.33-py2.py3-none-any.whl.metadata (3.0 kB)

Collecting jsonpointer==3.0.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 99))

  Downloading jsonpointer-3.0.0-py2.py3-none-any.whl.metadata (2.3 kB)

Collecting jsonschema==4.23.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 100))

  Downloading jsonschema-4.23.0-py3-none-any.whl.metadata (7.9 kB)

Collecting jsonschema-specifications==2024.10.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 101))

  Downloading jsonschema_specifications-2024.10.1-py3-none-any.whl.metadata (3.0 kB)

Collecting jupyter==1.1.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 102))

  Downloading jupyter-1.1.1-py2.py3-none-any.whl.metadata (2.0 kB)

Collecting jupyter-console==6.6.3 (from -r /mount/src/chatbot-gaib/requirements.txt (line 103))

  Downloading jupyter_console-6.6.3-py3-none-any.whl.metadata (5.8 kB)

Collecting jupyter-events==0.11.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 104))

  Downloading jupyter_events-0.11.0-py3-none-any.whl.metadata (5.8 kB)

Collecting jupyter-lsp==2.2.5 (from -r /mount/src/chatbot-gaib/requirements.txt (line 105))

  Downloading jupyter_lsp-2.2.5-py3-none-any.whl.metadata (1.8 kB)

Collecting jupyter_client==8.6.3 (from -r /mount/src/chatbot-gaib/requirements.txt (line 106))

  Downloading jupyter_client-8.6.3-py3-none-any.whl.metadata (8.3 kB)

Collecting jupyter_core==5.7.2 (from -r /mount/src/chatbot-gaib/requirements.txt (line 107))

  Downloading jupyter_core-5.7.2-py3-none-any.whl.metadata (3.4 kB)

Collecting jupyter_server==2.15.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 108))

  Downloading jupyter_server-2.15.0-py3-none-any.whl.metadata (8.4 kB)

Collecting jupyter_server_terminals==0.5.3 (from -r /mount/src/chatbot-gaib/requirements.txt (line 109))

  Downloading jupyter_server_terminals-0.5.3-py3-none-any.whl.metadata (5.6 kB)

Collecting jupyterlab==4.3.4 (from -r /mount/src/chatbot-gaib/requirements.txt (line 110))

  Downloading jupyterlab-4.3.4-py3-none-any.whl.metadata (16 kB)

Collecting jupyterlab_pygments==0.3.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 111))

  Downloading jupyterlab_pygments-0.3.0-py3-none-any.whl.metadata (4.4 kB)

Collecting jupyterlab_server==2.27.3 (from -r /mount/src/chatbot-gaib/requirements.txt (line 112))

  Downloading jupyterlab_server-2.27.3-py3-none-any.whl.metadata (5.9 kB)

Collecting jupyterlab_widgets==3.0.13 (from -r /mount/src/chatbot-gaib/requirements.txt (line 113))

  Downloading jupyterlab_widgets-3.0.13-py3-none-any.whl.metadata (4.1 kB)

Collecting kiwisolver==1.4.8 (from -r /mount/src/chatbot-gaib/requirements.txt (line 114))

  Downloading kiwisolver-1.4.8-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.2 kB)

Collecting kubernetes==33.1.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 115))

  Downloading kubernetes-33.1.0-py2.py3-none-any.whl.metadata (1.7 kB)

Collecting langchain==0.3.26 (from -r /mount/src/chatbot-gaib/requirements.txt (line 116))

  Downloading langchain-0.3.26-py3-none-any.whl.metadata (7.8 kB)

Collecting langchain-community==0.3.26 (from -r /mount/src/chatbot-gaib/requirements.txt (line 117))

  Downloading langchain_community-0.3.26-py3-none-any.whl.metadata (2.9 kB)

Collecting langchain-core==0.3.68 (from -r /mount/src/chatbot-gaib/requirements.txt (line 118))

  Downloading langchain_core-0.3.68-py3-none-any.whl.metadata (5.8 kB)

Collecting langchain-experimental==0.3.4 (from -r /mount/src/chatbot-gaib/requirements.txt (line 119))

  Downloading langchain_experimental-0.3.4-py3-none-any.whl.metadata (1.7 kB)

Collecting langchain-google-genai==2.1.5 (from -r /mount/src/chatbot-gaib/requirements.txt (line 120))

  Downloading langchain_google_genai-2.1.5-py3-none-any.whl.metadata (5.2 kB)

Collecting langchain-google-vertexai==2.0.27 (from -r /mount/src/chatbot-gaib/requirements.txt (line 121))

  Downloading langchain_google_vertexai-2.0.27-py3-none-any.whl.metadata (4.8 kB)

Collecting langchain-groq==0.3.2 (from -r /mount/src/chatbot-gaib/requirements.txt (line 122))

  Downloading langchain_groq-0.3.2-py3-none-any.whl.metadata (2.6 kB)

Collecting langchain-text-splitters==0.3.8 (from -r /mount/src/chatbot-gaib/requirements.txt (line 123))

  Downloading langchain_text_splitters-0.3.8-py3-none-any.whl.metadata (1.9 kB)

Collecting langsmith==0.3.45 (from -r /mount/src/chatbot-gaib/requirements.txt (line 124))

  Downloading langsmith-0.3.45-py3-none-any.whl.metadata (15 kB)

Collecting mando==0.7.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 125))

  Downloading mando-0.7.1-py2.py3-none-any.whl.metadata (7.4 kB)

Collecting markdown-it-py==3.0.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 126))

  Downloading markdown_it_py-3.0.0-py3-none-any.whl.metadata (6.9 kB)

Collecting MarkupSafe==3.0.2 (from -r /mount/src/chatbot-gaib/requirements.txt (line 127))

  Downloading MarkupSafe-3.0.2-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (4.0 kB)

Collecting marshmallow==3.26.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 128))

  Downloading marshmallow-3.26.1-py3-none-any.whl.metadata (7.3 kB)

Collecting matplotlib==3.10.3 (from -r /mount/src/chatbot-gaib/requirements.txt (line 129))

  Downloading matplotlib-3.10.3-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (11 kB)

Collecting matplotlib-inline==0.1.7 (from -r /mount/src/chatbot-gaib/requirements.txt (line 130))

  Downloading matplotlib_inline-0.1.7-py3-none-any.whl.metadata (3.9 kB)

Collecting mccabe==0.7.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 131))

  Downloading mccabe-0.7.0-py2.py3-none-any.whl.metadata (5.0 kB)

Collecting mdurl==0.1.2 (from -r /mount/src/chatbot-gaib/requirements.txt (line 132))

  Downloading mdurl-0.1.2-py3-none-any.whl.metadata (1.6 kB)

Collecting mistune==3.1.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 133))

  Downloading mistune-3.1.0-py3-none-any.whl.metadata (1.7 kB)

Collecting mmh3==5.1.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 134))

  Downloading mmh3-5.1.0-cp313-cp313-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (16 kB)

Collecting mpmath==1.3.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 135))

  Downloading mpmath-1.3.0-py3-none-any.whl.metadata (8.6 kB)

Collecting multidict==6.3.2 (from -r /mount/src/chatbot-gaib/requirements.txt (line 136))

  Downloading multidict-6.3.2-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (5.1 kB)

Collecting mypy_extensions==1.1.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 137))

  Downloading mypy_extensions-1.1.0-py3-none-any.whl.metadata (1.1 kB)

Collecting narwhals==1.43.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 138))

  Downloading narwhals-1.43.1-py3-none-any.whl.metadata (11 kB)

Collecting nbclient==0.10.2 (from -r /mount/src/chatbot-gaib/requirements.txt (line 139))

  Downloading nbclient-0.10.2-py3-none-any.whl.metadata (8.3 kB)

Collecting nbconvert==7.16.5 (from -r /mount/src/chatbot-gaib/requirements.txt (line 140))

  Downloading nbconvert-7.16.5-py3-none-any.whl.metadata (8.5 kB)

Collecting nbformat==5.10.4 (from -r /mount/src/chatbot-gaib/requirements.txt (line 141))

  Downloading nbformat-5.10.4-py3-none-any.whl.metadata (3.6 kB)

Collecting nest-asyncio==1.6.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 142))

  Downloading nest_asyncio-1.6.0-py3-none-any.whl.metadata (2.8 kB)

Collecting networkx==3.5 (from -r /mount/src/chatbot-gaib/requirements.txt (line 143))

  Downloading networkx-3.5-py3-none-any.whl.metadata (6.3 kB)

Collecting notebook==7.3.2 (from -r /mount/src/chatbot-gaib/requirements.txt (line 144))

  Downloading notebook-7.3.2-py3-none-any.whl.metadata (10 kB)

Collecting notebook_shim==0.2.4 (from -r /mount/src/chatbot-gaib/requirements.txt (line 145))

  Downloading notebook_shim-0.2.4-py3-none-any.whl.metadata (4.0 kB)

Collecting numexpr==2.11.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 146))

  Downloading numexpr-2.11.0-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (9.0 kB)

Collecting numpy==2.2.2 (from -r /mount/src/chatbot-gaib/requirements.txt (line 147))

  Downloading numpy-2.2.2-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (62 kB)

     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 62.0/62.0 kB 114.3 MB/s eta 0:00:00[2025-08-20 10:31:39.900142] 

Collecting oauthlib==3.3.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 148))

  Downloading oauthlib-3.3.1-py3-none-any.whl.metadata (7.9 kB)

Collecting onnxruntime==1.22.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 149))

  Downloading onnxruntime-1.22.1-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (4.9 kB)

Collecting openai==0.28.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 150))

  Downloading openai-0.28.0-py3-none-any.whl.metadata (13 kB)

Collecting openpyxl==3.1.5 (from -r /mount/src/chatbot-gaib/requirements.txt (line 151))

  Downloading openpyxl-3.1.5-py2.py3-none-any.whl.metadata (2.5 kB)

Collecting opentelemetry-api==1.35.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 152))

  Downloading opentelemetry_api-1.35.0-py3-none-any.whl.metadata (1.5 kB)

Collecting opentelemetry-exporter-otlp-proto-common==1.35.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 153))

  Downloading opentelemetry_exporter_otlp_proto_common-1.35.0-py3-none-any.whl.metadata (1.8 kB)

Collecting opentelemetry-exporter-otlp-proto-grpc==1.35.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 154))

  Downloading opentelemetry_exporter_otlp_proto_grpc-1.35.0-py3-none-any.whl.metadata (2.4 kB)

Collecting opentelemetry-proto==1.35.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 155))

  Downloading opentelemetry_proto-1.35.0-py3-none-any.whl.metadata (2.3 kB)

Collecting opentelemetry-sdk==1.35.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 156))

  Downloading opentelemetry_sdk-1.35.0-py3-none-any.whl.metadata (1.5 kB)

Collecting opentelemetry-semantic-conventions==0.56b0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 157))

  Downloading opentelemetry_semantic_conventions-0.56b0-py3-none-any.whl.metadata (2.4 kB)

Collecting oracledb==3.1.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 158))

  Downloading oracledb-3.1.0-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.manylinux_2_28_x86_64.whl.metadata (5.4 kB)

Collecting orjson==3.10.18 (from -r /mount/src/chatbot-gaib/requirements.txt (line 159))

  Downloading orjson-3.10.18-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (41 kB)

     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 41.9/41.9 kB 110.1 MB/s eta 0:00:00[2025-08-20 10:31:41.337066] 

Collecting overrides==7.7.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 160))

  Downloading overrides-7.7.0-py3-none-any.whl.metadata (5.8 kB)

Collecting packaging==24.2 (from -r /mount/src/chatbot-gaib/requirements.txt (line 161))

  Downloading packaging-24.2-py3-none-any.whl.metadata (3.2 kB)

Collecting pandas==2.3.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 162))

  Downloading pandas-2.3.1-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (91 kB)

     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 91.2/91.2 kB 180.2 MB/s eta 0:00:00[2025-08-20 10:31:41.682348] 

Collecting pandas-gbq==0.29.2 (from -r /mount/src/chatbot-gaib/requirements.txt (line 163))

  Downloading pandas_gbq-0.29.2-py3-none-any.whl.metadata (3.6 kB)

Collecting pandocfilters==1.5.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 164))

  Downloading pandocfilters-1.5.1-py2.py3-none-any.whl.metadata (9.0 kB)

Collecting parso==0.8.4 (from -r /mount/src/chatbot-gaib/requirements.txt (line 165))

  Downloading parso-0.8.4-py2.py3-none-any.whl.metadata (7.7 kB)

Collecting pathspec==0.12.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 166))

  Downloading pathspec-0.12.1-py3-none-any.whl.metadata (21 kB)

Collecting pillow==11.2.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 167))

  Downloading pillow-11.2.1-cp313-cp313-manylinux_2_28_x86_64.whl.metadata (8.9 kB)

Collecting pinecone-client==6.0.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 168))

  Downloading pinecone_client-6.0.0-py3-none-any.whl.metadata (3.4 kB)

Collecting pinecone-plugin-interface==0.0.7 (from -r /mount/src/chatbot-gaib/requirements.txt (line 169))

  Downloading pinecone_plugin_interface-0.0.7-py3-none-any.whl.metadata (1.2 kB)

Collecting platformdirs==4.3.6 (from -r /mount/src/chatbot-gaib/requirements.txt (line 170))

  Downloading platformdirs-4.3.6-py3-none-any.whl.metadata (11 kB)

Collecting plotly==6.2.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 171))

  Downloading plotly-6.2.0-py3-none-any.whl.metadata (8.5 kB)

Collecting posthog==5.4.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 172))

  Downloading posthog-5.4.0-py3-none-any.whl.metadata (5.7 kB)

Collecting prometheus_client==0.21.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 173))

  Downloading prometheus_client-0.21.1-py3-none-any.whl.metadata (1.8 kB)

Collecting prompt_toolkit==3.0.50 (from -r /mount/src/chatbot-gaib/requirements.txt (line 174))

  Downloading prompt_toolkit-3.0.50-py3-none-any.whl.metadata (6.6 kB)

Collecting propcache==0.3.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 175))

  Downloading propcache-0.3.1-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (10 kB)

Collecting proto-plus==1.26.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 176))

  Downloading proto_plus-1.26.1-py3-none-any.whl.metadata (2.2 kB)

Collecting protobuf==6.31.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 177))

  Downloading protobuf-6.31.1-cp39-abi3-manylinux2014_x86_64.whl.metadata (593 bytes)

Collecting psutil==6.1.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 178))

  Downloading psutil-6.1.1-cp36-abi3-manylinux_2_12_x86_64.manylinux2010_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (22 kB)

Collecting pure_eval==0.2.3 (from -r /mount/src/chatbot-gaib/requirements.txt (line 179))

  Downloading pure_eval-0.2.3-py3-none-any.whl.metadata (6.3 kB)

Collecting pyarrow==19.0.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 180))

  Downloading pyarrow-19.0.1-cp313-cp313-manylinux_2_28_x86_64.whl.metadata (3.3 kB)

Collecting pyasn1==0.6.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 181))

  Downloading pyasn1-0.6.1-py3-none-any.whl.metadata (8.4 kB)

Collecting pyasn1_modules==0.4.2 (from -r /mount/src/chatbot-gaib/requirements.txt (line 182))

  Downloading pyasn1_modules-0.4.2-py3-none-any.whl.metadata (3.5 kB)

Collecting pybase64==1.4.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 183))

  Downloading pybase64-1.4.1-cp313-cp313-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (8.4 kB)

Collecting pycparser==2.22 (from -r /mount/src/chatbot-gaib/requirements.txt (line 184))

  Downloading pycparser-2.22-py3-none-any.whl.metadata (943 bytes)

Collecting pydantic==2.11.2 (from -r /mount/src/chatbot-gaib/requirements.txt (line 185))

  Downloading pydantic-2.11.2-py3-none-any.whl.metadata (64 kB)

     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 64.7/64.7 kB 165.9 MB/s eta 0:00:00[2025-08-20 10:31:44.505513] 

Collecting pydantic-settings==2.10.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 186))

  Downloading pydantic_settings-2.10.0-py3-none-any.whl.metadata (3.4 kB)

Collecting pydantic_core==2.33.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 187))

  Downloading pydantic_core-2.33.1-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.8 kB)

Collecting pydata-google-auth==1.9.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 188))

  Downloading pydata_google_auth-1.9.1-py2.py3-none-any.whl.metadata (2.8 kB)

Collecting pydeck==0.9.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 189))

  Downloading pydeck-0.9.1-py2.py3-none-any.whl.metadata (4.1 kB)

Collecting Pygments==2.19.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 190))

  Downloading pygments-2.19.1-py3-none-any.whl.metadata (2.5 kB)

Collecting pylint==3.3.7 (from -r /mount/src/chatbot-gaib/requirements.txt (line 191))

  Downloading pylint-3.3.7-py3-none-any.whl.metadata (12 kB)

Collecting pyparsing==3.2.3 (from -r /mount/src/chatbot-gaib/requirements.txt (line 192))

  Downloading pyparsing-3.2.3-py3-none-any.whl.metadata (5.0 kB)

Collecting pypdf==5.8.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 193))

  Downloading pypdf-5.8.0-py3-none-any.whl.metadata (7.1 kB)

Collecting PyPika==0.48.9 (from -r /mount/src/chatbot-gaib/requirements.txt (line 194))

  Downloading PyPika-0.48.9.tar.gz (67 kB)

     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 67.3/67.3 kB 116.7 MB/s eta 0:00:00[2025-08-20 10:31:46.371962] 

  Installing build dependencies: started

  Installing build dependencies: finished with status 'done'

  Getting requirements to build wheel: started

  Getting requirements to build wheel: finished with status 'done'

  Preparing metadata (pyproject.toml): started

  Preparing metadata (pyproject.toml): finished with status 'done'

Collecting pyproject_hooks==1.2.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 195))

  Downloading pyproject_hooks-1.2.0-py3-none-any.whl.metadata (1.3 kB)

Collecting pyreadline3==3.5.4 (from -r /mount/src/chatbot-gaib/requirements.txt (line 196))

  Downloading pyreadline3-3.5.4-py3-none-any.whl.metadata (4.7 kB)

Collecting python-dateutil==2.9.0.post0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 197))

  Downloading python_dateutil-2.9.0.post0-py2.py3-none-any.whl.metadata (8.4 kB)

Collecting python-dotenv==1.1.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 198))

  Downloading python_dotenv-1.1.0-py3-none-any.whl.metadata (24 kB)

Collecting python-json-logger==3.2.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 199))

  Downloading python_json_logger-3.2.1-py3-none-any.whl.metadata (4.1 kB)

Collecting pytz==2024.2 (from -r /mount/src/chatbot-gaib/requirements.txt (line 200))

  Downloading pytz-2024.2-py2.py3-none-any.whl.metadata (22 kB)

Collecting pywinpty==2.0.14 (from -r /mount/src/chatbot-gaib/requirements.txt (line 201))

  Downloading pywinpty-2.0.14.tar.gz (27 kB)

  Installing build dependencies: started

  Installing build dependencies: finished with status 'done'

  Getting requirements to build wheel: started

  Getting requirements to build wheel: finished with status 'done'

  Installing backend dependencies: started

  Installing backend dependencies: finished with status 'done'

  Preparing metadata (pyproject.toml): started

  Preparing metadata (pyproject.toml): finished with status 'done'

Collecting PyYAML==6.0.2 (from -r /mount/src/chatbot-gaib/requirements.txt (line 202))

  Downloading PyYAML-6.0.2-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (2.1 kB)

Collecting pyzmq==26.2.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 203))

  Downloading pyzmq-26.2.0-cp313-cp313-manylinux_2_28_x86_64.whl.metadata (6.2 kB)

Collecting radon==6.0.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 204))

  Downloading radon-6.0.1-py2.py3-none-any.whl.metadata (8.2 kB)

Collecting referencing==0.36.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 205))

  Downloading referencing-0.36.1-py3-none-any.whl.metadata (2.8 kB)

Collecting regex==2024.11.6 (from -r /mount/src/chatbot-gaib/requirements.txt (line 206))

  Downloading regex-2024.11.6-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (40 kB)

     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 40.5/40.5 kB 134.0 MB/s eta 0:00:00[2025-08-20 10:31:57.511815] 

Collecting requests==2.32.4 (from -r /mount/src/chatbot-gaib/requirements.txt (line 207))

  Downloading requests-2.32.4-py3-none-any.whl.metadata (4.9 kB)

Collecting requests-oauthlib==2.0.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 208))

  Downloading requests_oauthlib-2.0.0-py2.py3-none-any.whl.metadata (11 kB)

Collecting requests-toolbelt==1.0.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 209))

  Downloading requests_toolbelt-1.0.0-py2.py3-none-any.whl.metadata (14 kB)

Collecting rfc3339-validator==0.1.4 (from -r /mount/src/chatbot-gaib/requirements.txt (line 210))

  Downloading rfc3339_validator-0.1.4-py2.py3-none-any.whl.metadata (1.5 kB)

Collecting rfc3986-validator==0.1.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 211))

  Downloading rfc3986_validator-0.1.1-py2.py3-none-any.whl.metadata (1.7 kB)

Collecting rich==14.0.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 212))

  Downloading rich-14.0.0-py3-none-any.whl.metadata (18 kB)

Collecting rpds-py==0.22.3 (from -r /mount/src/chatbot-gaib/requirements.txt (line 213))

  Downloading rpds_py-0.22.3-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (4.2 kB)

Collecting rsa==4.9.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 214))

  Downloading rsa-4.9.1-py3-none-any.whl.metadata (5.6 kB)

Collecting safetensors==0.5.3 (from -r /mount/src/chatbot-gaib/requirements.txt (line 215))

  Downloading safetensors-0.5.3-cp38-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (3.8 kB)

Collecting scikit-learn==1.7.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 216))

  Downloading scikit_learn-1.7.1-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (11 kB)

Collecting scipy==1.16.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 217))

  Downloading scipy-1.16.0-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (61 kB)

     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 61.9/61.9 kB 161.8 MB/s eta 0:00:00[2025-08-20 10:31:59.256276] 

Collecting seaborn==0.13.2 (from -r /mount/src/chatbot-gaib/requirements.txt (line 218))

  Downloading seaborn-0.13.2-py3-none-any.whl.metadata (5.4 kB)

Collecting Send2Trash==1.8.3 (from -r /mount/src/chatbot-gaib/requirements.txt (line 219))

  Downloading Send2Trash-1.8.3-py3-none-any.whl.metadata (4.0 kB)

Collecting sentence-transformers==5.0.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 220))

  Downloading sentence_transformers-5.0.0-py3-none-any.whl.metadata (16 kB)

Collecting setuptools==75.8.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 221))

  Downloading setuptools-75.8.0-py3-none-any.whl.metadata (6.7 kB)

Collecting shapely==2.1.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 222))

  Downloading shapely-2.1.1-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.8 kB)

Collecting shellingham==1.5.4 (from -r /mount/src/chatbot-gaib/requirements.txt (line 223))

  Downloading shellingham-1.5.4-py2.py3-none-any.whl.metadata (3.5 kB)

Collecting six==1.17.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 224))

  Downloading six-1.17.0-py2.py3-none-any.whl.metadata (1.7 kB)

Collecting smmap==5.0.2 (from -r /mount/src/chatbot-gaib/requirements.txt (line 225))

  Downloading smmap-5.0.2-py3-none-any.whl.metadata (4.3 kB)

Collecting sniffio==1.3.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 226))

  Downloading sniffio-1.3.1-py3-none-any.whl.metadata (3.9 kB)

Collecting soupsieve==2.6 (from -r /mount/src/chatbot-gaib/requirements.txt (line 227))

  Downloading soupsieve-2.6-py3-none-any.whl.metadata (4.6 kB)

Collecting SQLAlchemy==2.0.41 (from -r /mount/src/chatbot-gaib/requirements.txt (line 228))

  Downloading sqlalchemy-2.0.41-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (9.6 kB)

Collecting sqlalchemy_bigquery==0.0.7 (from -r /mount/src/chatbot-gaib/requirements.txt (line 229))

  Downloading sqlalchemy_bigquery-0.0.7.tar.gz (5.1 kB)

  Installing build dependencies: started

  Installing build dependencies: finished with status 'done'

  Getting requirements to build wheel: started

  Getting requirements to build wheel: finished with status 'done'

  Preparing metadata (pyproject.toml): started

  Preparing metadata (pyproject.toml): finished with status 'done'

Collecting sqlparse==0.5.3 (from -r /mount/src/chatbot-gaib/requirements.txt (line 230))

  Downloading sqlparse-0.5.3-py3-none-any.whl.metadata (3.9 kB)

Collecting stack-data==0.6.3 (from -r /mount/src/chatbot-gaib/requirements.txt (line 231))

  Downloading stack_data-0.6.3-py3-none-any.whl.metadata (18 kB)

Collecting streamlit==1.46.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 232))

  Downloading streamlit-1.46.0-py3-none-any.whl.metadata (9.0 kB)

Collecting sympy==1.14.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 233))

  Downloading sympy-1.14.0-py3-none-any.whl.metadata (12 kB)

Collecting tabulate==0.9.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 234))

  Downloading tabulate-0.9.0-py3-none-any.whl.metadata (34 kB)

Collecting tenacity==8.5.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 235))

  Downloading tenacity-8.5.0-py3-none-any.whl.metadata (1.2 kB)

Collecting terminado==0.18.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 236))

  Downloading terminado-0.18.1-py3-none-any.whl.metadata (5.8 kB)

Collecting threadpoolctl==3.6.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 237))

  Downloading threadpoolctl-3.6.0-py3-none-any.whl.metadata (13 kB)

Collecting tinycss2==1.4.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 238))

  Downloading tinycss2-1.4.0-py3-none-any.whl.metadata (3.0 kB)

Collecting tokenizers==0.21.2 (from -r /mount/src/chatbot-gaib/requirements.txt (line 239))

  Downloading tokenizers-0.21.2-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.8 kB)

Collecting toml==0.10.2 (from -r /mount/src/chatbot-gaib/requirements.txt (line 240))

  Downloading toml-0.10.2-py2.py3-none-any.whl.metadata (7.1 kB)

Collecting tomlkit==0.13.3 (from -r /mount/src/chatbot-gaib/requirements.txt (line 241))

  Downloading tomlkit-0.13.3-py3-none-any.whl.metadata (2.8 kB)

Collecting torch==2.7.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 242))

  Downloading torch-2.7.1-cp313-cp313-manylinux_2_28_x86_64.whl.metadata (29 kB)

Collecting tornado==6.4.2 (from -r /mount/src/chatbot-gaib/requirements.txt (line 243))

  Downloading tornado-6.4.2-cp38-abi3-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (2.5 kB)

Collecting tqdm==4.67.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 244))

  Downloading tqdm-4.67.1-py3-none-any.whl.metadata (57 kB)

     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 57.7/57.7 kB 135.7 MB/s eta 0:00:00[2025-08-20 10:32:05.098441] 

Collecting traitlets==5.14.3 (from -r /mount/src/chatbot-gaib/requirements.txt (line 245))

  Downloading traitlets-5.14.3-py3-none-any.whl.metadata (10 kB)

Collecting transformers==4.53.2 (from -r /mount/src/chatbot-gaib/requirements.txt (line 246))

  Downloading transformers-4.53.2-py3-none-any.whl.metadata (40 kB)

     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 40.9/40.9 kB 164.0 MB/s eta 0:00:00[2025-08-20 10:32:05.247898] 

Collecting typer==0.16.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 247))

  Downloading typer-0.16.0-py3-none-any.whl.metadata (15 kB)

Collecting types-python-dateutil==2.9.0.20241206 (from -r /mount/src/chatbot-gaib/requirements.txt (line 248))

  Downloading types_python_dateutil-2.9.0.20241206-py3-none-any.whl.metadata (2.1 kB)

Collecting typing-inspect==0.9.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 249))

  Downloading typing_inspect-0.9.0-py3-none-any.whl.metadata (1.5 kB)

Collecting typing-inspection==0.4.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 250))

  Downloading typing_inspection-0.4.0-py3-none-any.whl.metadata (2.6 kB)

Collecting typing_extensions==4.12.2 (from -r /mount/src/chatbot-gaib/requirements.txt (line 251))

  Downloading typing_extensions-4.12.2-py3-none-any.whl.metadata (3.0 kB)

Collecting tzdata==2025.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 252))

  Downloading tzdata-2025.1-py2.py3-none-any.whl.metadata (1.4 kB)

Collecting uri-template==1.3.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 253))

  Downloading uri_template-1.3.0-py3-none-any.whl.metadata (8.8 kB)

Collecting urllib3==2.5.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 254))

  Downloading urllib3-2.5.0-py3-none-any.whl.metadata (6.5 kB)

Collecting uvicorn==0.35.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 255))

  Downloading uvicorn-0.35.0-py3-none-any.whl.metadata (6.5 kB)

Collecting validators==0.35.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 256))

  Downloading validators-0.35.0-py3-none-any.whl.metadata (3.9 kB)

Collecting watchdog==6.0.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 257))

  Downloading watchdog-6.0.0-py3-none-manylinux2014_x86_64.whl.metadata (44 kB)

     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 44.3/44.3 kB 111.5 MB/s eta 0:00:00[2025-08-20 10:32:05.852171] 

Collecting watchfiles==1.1.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 258))

  Downloading watchfiles-1.1.0-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (4.9 kB)

Collecting wcwidth==0.2.13 (from -r /mount/src/chatbot-gaib/requirements.txt (line 259))

  Downloading wcwidth-0.2.13-py2.py3-none-any.whl.metadata (14 kB)

Collecting webcolors==24.11.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 260))

  Downloading webcolors-24.11.1-py3-none-any.whl.metadata (2.2 kB)

Collecting webencodings==0.5.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 261))

  Downloading webencodings-0.5.1-py2.py3-none-any.whl.metadata (2.1 kB)

Collecting websocket-client==1.8.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 262))

  Downloading websocket_client-1.8.0-py3-none-any.whl.metadata (8.0 kB)

Collecting websockets==15.0.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 263))

  Downloading websockets-15.0.1-cp313-cp313-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.8 kB)

Collecting Werkzeug==3.1.3 (from -r /mount/src/chatbot-gaib/requirements.txt (line 264))

  Downloading werkzeug-3.1.3-py3-none-any.whl.metadata (3.7 kB)

Collecting widgetsnbextension==4.0.13 (from -r /mount/src/chatbot-gaib/requirements.txt (line 265))

  Downloading widgetsnbextension-4.0.13-py3-none-any.whl.metadata (1.6 kB)

Collecting yarl==1.19.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 266))

  Downloading yarl-1.19.0-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (71 kB)

     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 71.8/71.8 kB 139.2 MB/s eta 0:00:00[2025-08-20 10:32:06.845303] 

Collecting zipp==3.23.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 267))

  Downloading zipp-3.23.0-py3-none-any.whl.metadata (3.6 kB)

Collecting zstandard==0.23.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 268))

  Downloading zstandard-0.23.0-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (3.0 kB)

Collecting hf-xet<2.0.0,>=1.1.2 (from huggingface-hub==0.33.4->-r /mount/src/chatbot-gaib/requirements.txt (line 82))

  Downloading hf_xet-1.1.8-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (703 bytes)

Collecting pexpect>4.3 (from ipython==8.31.0->-r /mount/src/chatbot-gaib/requirements.txt (line 88))

  Downloading pexpect-4.9.0-py2.py3-none-any.whl.metadata (2.5 kB)

Collecting ptyprocess (from terminado==0.18.1->-r /mount/src/chatbot-gaib/requirements.txt (line 236))

  Downloading ptyprocess-0.7.0-py2.py3-none-any.whl.metadata (1.3 kB)

Collecting nvidia-cuda-nvrtc-cu12==12.6.77 (from torch==2.7.1->-r /mount/src/chatbot-gaib/requirements.txt (line 242))

  Downloading nvidia_cuda_nvrtc_cu12-12.6.77-py3-none-manylinux2014_x86_64.whl.metadata (1.5 kB)

Collecting nvidia-cuda-runtime-cu12==12.6.77 (from torch==2.7.1->-r /mount/src/chatbot-gaib/requirements.txt (line 242))

  Downloading nvidia_cuda_runtime_cu12-12.6.77-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.5 kB)

Collecting nvidia-cuda-cupti-cu12==12.6.80 (from torch==2.7.1->-r /mount/src/chatbot-gaib/requirements.txt (line 242))

  Downloading nvidia_cuda_cupti_cu12-12.6.80-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.6 kB)

Collecting nvidia-cudnn-cu12==9.5.1.17 (from torch==2.7.1->-r /mount/src/chatbot-gaib/requirements.txt (line 242))

  Downloading nvidia_cudnn_cu12-9.5.1.17-py3-none-manylinux_2_28_x86_64.whl.metadata (1.6 kB)

Collecting nvidia-cublas-cu12==12.6.4.1 (from torch==2.7.1->-r /mount/src/chatbot-gaib/requirements.txt (line 242))

  Downloading nvidia_cublas_cu12-12.6.4.1-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.5 kB)

Collecting nvidia-cufft-cu12==11.3.0.4 (from torch==2.7.1->-r /mount/src/chatbot-gaib/requirements.txt (line 242))

  Downloading nvidia_cufft_cu12-11.3.0.4-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.5 kB)

Collecting nvidia-curand-cu12==10.3.7.77 (from torch==2.7.1->-r /mount/src/chatbot-gaib/requirements.txt (line 242))

  Downloading nvidia_curand_cu12-10.3.7.77-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.5 kB)

Collecting nvidia-cusolver-cu12==11.7.1.2 (from torch==2.7.1->-r /mount/src/chatbot-gaib/requirements.txt (line 242))

  Downloading nvidia_cusolver_cu12-11.7.1.2-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.6 kB)

Collecting nvidia-cusparse-cu12==12.5.4.2 (from torch==2.7.1->-r /mount/src/chatbot-gaib/requirements.txt (line 242))

  Downloading nvidia_cusparse_cu12-12.5.4.2-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.6 kB)

Collecting nvidia-cusparselt-cu12==0.6.3 (from torch==2.7.1->-r /mount/src/chatbot-gaib/requirements.txt (line 242))

  Downloading nvidia_cusparselt_cu12-0.6.3-py3-none-manylinux2014_x86_64.whl.metadata (6.8 kB)

Collecting nvidia-nccl-cu12==2.26.2 (from torch==2.7.1->-r /mount/src/chatbot-gaib/requirements.txt (line 242))

  Downloading nvidia_nccl_cu12-2.26.2-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (2.0 kB)

Collecting nvidia-nvtx-cu12==12.6.77 (from torch==2.7.1->-r /mount/src/chatbot-gaib/requirements.txt (line 242))

  Downloading nvidia_nvtx_cu12-12.6.77-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.6 kB)

Collecting nvidia-nvjitlink-cu12==12.6.85 (from torch==2.7.1->-r /mount/src/chatbot-gaib/requirements.txt (line 242))

  Downloading nvidia_nvjitlink_cu12-12.6.85-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl.metadata (1.5 kB)

Collecting nvidia-cufile-cu12==1.11.1.6 (from torch==2.7.1->-r /mount/src/chatbot-gaib/requirements.txt (line 242))

  Downloading nvidia_cufile_cu12-1.11.1.6-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.5 kB)

Collecting triton==3.3.1 (from torch==2.7.1->-r /mount/src/chatbot-gaib/requirements.txt (line 242))

  Downloading triton-3.3.1-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (1.5 kB)

Collecting uvloop>=0.15.1 (from uvicorn[standard]>=0.18.3->chromadb==1.0.15->-r /mount/src/chatbot-gaib/requirements.txt (line 27))

  Downloading uvloop-0.21.0-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (4.9 kB)

WARNING: The candidate selected for download or install is a yanked version: 'multidict' candidate (version 6.3.2 at https://files.pythonhosted.org/packages/28/f8/18c81f5c5b7453dd8d15dc61ceca23d03c55e69f1937842039be2d8c4428/multidict-6.3.2-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (from https://pypi.org/simple/multidict/) (requires-python:>=3.9))

Reason for being yanked: Memory leak: https://github.com/aio-libs/multidict/issues/1131#issuecomment-2787514071

Downloading aiohappyeyeballs-2.6.1-py3-none-any.whl (15 kB)

Downloading aiohttp-3.11.16-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.7 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.7/1.7 MB 87.2 MB/s eta 0:00:00[2025-08-20 10:32:30.589055] 

Downloading aiosignal-1.3.2-py2.py3-none-any.whl (7.6 kB)

Downloading altair-5.5.0-py3-none-any.whl (731 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 731.2/731.2 kB 166.3 MB/s eta 0:00:00[2025-08-20 10:32:30.612812] 

Downloading annotated_types-0.7.0-py3-none-any.whl (13 kB)

Downloading anyio-4.8.0-py3-none-any.whl (96 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 96.0/96.0 kB 176.5 MB/s eta 0:00:00[2025-08-20 10:32:30.632664] 

Downloading argon2_cffi-23.1.0-py3-none-any.whl (15 kB)

Downloading argon2_cffi_bindings-21.2.0-cp36-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (86 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 86.2/86.2 kB 169.4 MB/s eta 0:00:00[2025-08-20 10:32:30.652549] 

Downloading arrow-1.3.0-py3-none-any.whl (66 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 66.4/66.4 kB 178.4 MB/s eta 0:00:00[2025-08-20 10:32:30.663896] 

Downloading astroid-3.3.10-py3-none-any.whl (275 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 275.4/275.4 kB 213.6 MB/s eta 0:00:00[2025-08-20 10:32:30.676928] 

Downloading asttokens-3.0.0-py3-none-any.whl (26 kB)

Downloading async_lru-2.0.4-py3-none-any.whl (6.1 kB)

Downloading attrs-24.3.0-py3-none-any.whl (63 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 63.4/63.4 kB 150.7 MB/s eta 0:00:00[2025-08-20 10:32:30.706212] 

Downloading babel-2.16.0-py3-none-any.whl (9.6 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 9.6/9.6 MB 128.9 MB/s eta 0:00:00[2025-08-20 10:32:30.796708] 

Downloading backoff-2.2.1-py3-none-any.whl (15 kB)

Downloading bcrypt-4.3.0-cp39-abi3-manylinux_2_28_x86_64.whl (284 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 284.5/284.5 kB 131.0 MB/s eta 0:00:00[2025-08-20 10:32:30.821532] 

Downloading beautifulsoup4-4.12.3-py3-none-any.whl (147 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 147.9/147.9 kB 183.8 MB/s eta 0:00:00[2025-08-20 10:32:30.835188] 

Downloading black-25.1.0-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.manylinux_2_28_x86_64.whl (1.8 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.8/1.8 MB 124.5 MB/s eta 0:00:00[2025-08-20 10:32:30.873714] 

Downloading bleach-6.2.0-py3-none-any.whl (163 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 163.4/163.4 kB 151.7 MB/s eta 0:00:00[2025-08-20 10:32:30.886804] 

Downloading blinker-1.9.0-py3-none-any.whl (8.5 kB)

Downloading bottleneck-1.5.0-cp313-cp313-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (362 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 362.8/362.8 kB 149.0 MB/s eta 0:00:00[2025-08-20 10:32:30.913403] 

Downloading build-1.2.2.post1-py3-none-any.whl (22 kB)

Downloading cachetools-5.5.2-py3-none-any.whl (10 kB)

Downloading certifi-2025.6.15-py3-none-any.whl (157 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 157.7/157.7 kB 157.8 MB/s eta 0:00:00[2025-08-20 10:32:30.943908] 

Downloading cffi-1.17.1-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (479 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 479.4/479.4 kB 189.6 MB/s eta 0:00:00[2025-08-20 10:32:30.958074] 

Downloading charset_normalizer-3.4.2-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (148 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 148.2/148.2 kB 143.4 MB/s eta 0:00:00[2025-08-20 10:32:30.971908] 

Downloading chromadb-1.0.15-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (19.5 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 19.5/19.5 MB 194.2 MB/s eta 0:00:00[2025-08-20 10:32:31.088989] 

Downloading click-8.1.8-py3-none-any.whl (98 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 98.2/98.2 kB 174.4 MB/s eta 0:00:00[2025-08-20 10:32:31.100093] 

Downloading colorama-0.4.6-py2.py3-none-any.whl (25 kB)

Downloading coloredlogs-15.0.1-py2.py3-none-any.whl (46 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 46.0/46.0 kB 129.6 MB/s eta 0:00:00[2025-08-20 10:32:31.119763] 

Downloading comm-0.2.2-py3-none-any.whl (7.2 kB)

Downloading contourpy-1.3.2-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (322 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 322.7/322.7 kB 206.0 MB/s eta 0:00:00[2025-08-20 10:32:31.143049] 

Downloading cryptography-44.0.2-cp39-abi3-manylinux_2_28_x86_64.whl (4.2 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.2/4.2 MB 184.5 MB/s eta 0:00:00[2025-08-20 10:32:31.179630] 

Downloading cycler-0.12.1-py3-none-any.whl (8.3 kB)

Downloading dataclasses_json-0.6.7-py3-none-any.whl (28 kB)

Downloading db_dtypes-1.4.3-py3-none-any.whl (18 kB)

Downloading debugpy-1.8.12-cp313-cp313-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (4.2 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.2/4.2 MB 210.5 MB/s eta 0:00:00[2025-08-20 10:32:31.240718] 

Downloading decorator-5.1.1-py3-none-any.whl (9.1 kB)

Downloading defusedxml-0.7.1-py2.py3-none-any.whl (25 kB)

Downloading dill-0.4.0-py3-none-any.whl (119 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 119.7/119.7 kB 132.5 MB/s eta 0:00:00[2025-08-20 10:32:31.270081] 

Downloading distro-1.9.0-py3-none-any.whl (20 kB)

Downloading docstring_parser-0.16-py3-none-any.whl (36 kB)

Downloading durationpy-0.10-py3-none-any.whl (3.9 kB)

Downloading et_xmlfile-2.0.0-py3-none-any.whl (18 kB)

Downloading executing-2.2.0-py2.py3-none-any.whl (26 kB)

Downloading Faker-35.0.0-py3-none-any.whl (1.9 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.9/1.9 MB 143.8 MB/s eta 0:00:00[2025-08-20 10:32:31.340618] 

Downloading fastjsonschema-2.21.1-py3-none-any.whl (23 kB)

Downloading filelock-3.18.0-py3-none-any.whl (16 kB)

Downloading filetype-1.2.0-py2.py3-none-any.whl (19 kB)

Downloading flask-3.1.0-py3-none-any.whl (102 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 103.0/103.0 kB 121.9 MB/s eta 0:00:00[2025-08-20 10:32:31.381654] 

Downloading flatbuffers-25.2.10-py2.py3-none-any.whl (30 kB)

Downloading fonttools-4.59.0-cp313-cp313-manylinux1_x86_64.manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_5_x86_64.whl (4.9 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.9/4.9 MB 128.8 MB/s eta 0:00:00[2025-08-20 10:32:31.444925] 

Downloading fqdn-1.5.1-py3-none-any.whl (9.1 kB)

Downloading frozenlist-1.5.0-cp313-cp313-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (267 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 267.0/267.0 kB 134.1 MB/s eta 0:00:00[2025-08-20 10:32:31.469110] 

Downloading fsspec-2025.7.0-py3-none-any.whl (199 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 199.6/199.6 kB 134.8 MB/s eta 0:00:00[2025-08-20 10:32:31.483284] 

Downloading gitdb-4.0.12-py3-none-any.whl (62 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 62.8/62.8 kB 117.7 MB/s eta 0:00:00[2025-08-20 10:32:31.496181] 

Downloading GitPython-3.1.44-py3-none-any.whl (207 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 207.6/207.6 kB 138.2 MB/s eta 0:00:00[2025-08-20 10:32:31.510021] 

Downloading google_ai_generativelanguage-0.6.18-py3-none-any.whl (1.4 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.4/1.4 MB 152.3 MB/s eta 0:00:00[2025-08-20 10:32:31.533814] 

Downloading google_api_core-2.25.1-py3-none-any.whl (160 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 160.8/160.8 kB 130.9 MB/s eta 0:00:00[2025-08-20 10:32:31.547678] 

Downloading google_auth-2.40.3-py2.py3-none-any.whl (216 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 216.1/216.1 kB 122.9 MB/s eta 0:00:00[2025-08-20 10:32:31.561915] 

Downloading google_auth_oauthlib-1.2.2-py3-none-any.whl (19 kB)

Downloading google_cloud_aiplatform-1.105.0-py2.py3-none-any.whl (7.9 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7.9/7.9 MB 142.1 MB/s eta 0:00:00[2025-08-20 10:32:31.641622] 

Downloading google_cloud_bigquery-3.34.0-py3-none-any.whl (253 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 253.6/253.6 kB 162.0 MB/s eta 0:00:00[2025-08-20 10:32:31.656730] 

Downloading google_cloud_bigquery_storage-2.32.0-py3-none-any.whl (296 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 296.4/296.4 kB 179.0 MB/s eta 0:00:00[2025-08-20 10:32:31.671610] 

Downloading google_cloud_core-2.4.3-py2.py3-none-any.whl (29 kB)

Downloading google_cloud_resource_manager-1.14.2-py3-none-any.whl (394 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 394.3/394.3 kB 205.4 MB/s eta 0:00:00[2025-08-20 10:32:31.697348] 

Downloading google_cloud_storage-2.19.0-py2.py3-none-any.whl (131 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 131.8/131.8 kB 142.6 MB/s eta 0:00:00[2025-08-20 10:32:31.710426] 

Downloading google_crc32c-1.7.1-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (32 kB)

Downloading google_genai-1.21.1-py3-none-any.whl (206 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 206.4/206.4 kB 215.1 MB/s eta 0:00:00[2025-08-20 10:32:31.732880] 

Downloading google_resumable_media-2.7.2-py2.py3-none-any.whl (81 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 81.3/81.3 kB 140.0 MB/s eta 0:00:00[2025-08-20 10:32:31.744533] 

Downloading googleapis_common_protos-1.70.0-py3-none-any.whl (294 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 294.5/294.5 kB 206.1 MB/s eta 0:00:00[2025-08-20 10:32:31.756610] 

Downloading greenlet-3.2.3-cp313-cp313-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl (608 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 608.4/608.4 kB 132.6 MB/s eta 0:00:00[2025-08-20 10:32:31.776613] 

Downloading groq-0.28.0-py3-none-any.whl (130 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 130.2/130.2 kB 181.6 MB/s eta 0:00:00[2025-08-20 10:32:31.790417] 

Downloading grpc_google_iam_v1-0.14.2-py3-none-any.whl (19 kB)

Downloading grpcio-1.73.0-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (6.0 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 6.0/6.0 MB 155.1 MB/s eta 0:00:00[2025-08-20 10:32:31.856271] 

Downloading grpcio_status-1.73.0-py3-none-any.whl (14 kB)

Downloading h11-0.14.0-py3-none-any.whl (58 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 58.3/58.3 kB 111.4 MB/s eta 0:00:00[2025-08-20 10:32:31.879169] 

Downloading httpcore-1.0.7-py3-none-any.whl (78 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 78.6/78.6 kB 119.6 MB/s eta 0:00:00[2025-08-20 10:32:31.891979] 

Downloading httptools-0.6.4-cp313-cp313-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (473 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 473.8/473.8 kB 129.6 MB/s eta 0:00:00[2025-08-20 10:32:31.914723] 

Downloading httpx-0.28.1-py3-none-any.whl (73 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 73.5/73.5 kB 123.6 MB/s eta 0:00:00[2025-08-20 10:32:31.927765] 

Downloading httpx_sse-0.4.0-py3-none-any.whl (7.8 kB)

Downloading huggingface_hub-0.33.4-py3-none-any.whl (515 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 515.3/515.3 kB 132.3 MB/s eta 0:00:00[2025-08-20 10:32:31.954338] 

Downloading humanfriendly-10.0-py2.py3-none-any.whl (86 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 86.8/86.8 kB 121.7 MB/s eta 0:00:00[2025-08-20 10:32:31.966721] 

Downloading idna-3.10-py3-none-any.whl (70 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 70.4/70.4 kB 120.4 MB/s eta 0:00:00[2025-08-20 10:32:31.978529] 

Downloading importlib_metadata-8.7.0-py3-none-any.whl (27 kB)

Downloading importlib_resources-6.5.2-py3-none-any.whl (37 kB)

Downloading ipykernel-6.29.5-py3-none-any.whl (117 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 117.2/117.2 kB 128.1 MB/s eta 0:00:00[2025-08-20 10:32:32.009288] 

Downloading ipython-8.31.0-py3-none-any.whl (821 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 821.6/821.6 kB 135.4 MB/s eta 0:00:00[2025-08-20 10:32:32.031062] 

Downloading ipywidgets-8.1.5-py3-none-any.whl (139 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 139.8/139.8 kB 130.6 MB/s eta 0:00:00[2025-08-20 10:32:32.045185] 

Downloading isoduration-20.11.0-py3-none-any.whl (11 kB)

Downloading isort-6.0.1-py3-none-any.whl (94 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 94.2/94.2 kB 128.3 MB/s eta 0:00:00[2025-08-20 10:32:32.066470] 

Downloading itsdangerous-2.2.0-py3-none-any.whl (16 kB)

Downloading jedi-0.19.2-py2.py3-none-any.whl (1.6 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.6/1.6 MB 125.9 MB/s eta 0:00:00[2025-08-20 10:32:32.100978] 

Downloading jinja2-3.1.5-py3-none-any.whl (134 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 134.6/134.6 kB 215.5 MB/s eta 0:00:00[2025-08-20 10:32:32.113992] 

Downloading jiter-0.9.0-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (352 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 352.4/352.4 kB 221.0 MB/s eta 0:00:00[2025-08-20 10:32:32.128554] 

Downloading joblib-1.5.1-py3-none-any.whl (307 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 307.7/307.7 kB 151.8 MB/s eta 0:00:00[2025-08-20 10:32:32.142282] 

Downloading json5-0.10.0-py3-none-any.whl (34 kB)

Downloading jsonpatch-1.33-py2.py3-none-any.whl (12 kB)

Downloading jsonpointer-3.0.0-py2.py3-none-any.whl (7.6 kB)

Downloading jsonschema-4.23.0-py3-none-any.whl (88 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 88.5/88.5 kB 125.8 MB/s eta 0:00:00[2025-08-20 10:32:32.182755] 

Downloading jsonschema_specifications-2024.10.1-py3-none-any.whl (18 kB)

Downloading jupyter-1.1.1-py2.py3-none-any.whl (2.7 kB)

Downloading jupyter_console-6.6.3-py3-none-any.whl (24 kB)

Downloading jupyter_events-0.11.0-py3-none-any.whl (19 kB)

Downloading jupyter_lsp-2.2.5-py3-none-any.whl (69 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 69.1/69.1 kB 132.2 MB/s eta 0:00:00

Downloading jupyter_client-8.6.3-py3-none-any.whl (106 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 106.1/106.1 kB 185.9 MB/s eta 0:00:00[2025-08-20 10:32:32.254108] 

Downloading jupyter_core-5.7.2-py3-none-any.whl (28 kB)

Downloading jupyter_server-2.15.0-py3-none-any.whl (385 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 385.8/385.8 kB 206.3 MB/s eta 0:00:00[2025-08-20 10:32:32.277763] 

Downloading jupyter_server_terminals-0.5.3-py3-none-any.whl (13 kB)

Downloading jupyterlab-4.3.4-py3-none-any.whl (11.7 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 11.7/11.7 MB 173.8 MB/s eta 0:00:00[2025-08-20 10:32:32.369985] 

Downloading jupyterlab_pygments-0.3.0-py3-none-any.whl (15 kB)

Downloading jupyterlab_server-2.27.3-py3-none-any.whl (59 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 59.7/59.7 kB 120.2 MB/s eta 0:00:00[2025-08-20 10:32:32.390132] 

Downloading jupyterlab_widgets-3.0.13-py3-none-any.whl (214 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 214.4/214.4 kB 180.9 MB/s eta 0:00:00[2025-08-20 10:32:32.403661] 

Downloading kiwisolver-1.4.8-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.5 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.5/1.5 MB 202.3 MB/s eta 0:00:00[2025-08-20 10:32:32.425024] 

Downloading kubernetes-33.1.0-py2.py3-none-any.whl (1.9 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.9/1.9 MB 208.4 MB/s eta 0:00:00[2025-08-20 10:32:32.446040] 

Downloading langchain-0.3.26-py3-none-any.whl (1.0 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.0/1.0 MB 134.6 MB/s eta 0:00:00[2025-08-20 10:32:32.469579] 

Downloading langchain_community-0.3.26-py3-none-any.whl (2.5 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.5/2.5 MB 142.4 MB/s eta 0:00:00[2025-08-20 10:32:32.506638] 

Downloading langchain_core-0.3.68-py3-none-any.whl (441 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 441.4/441.4 kB 159.6 MB/s eta 0:00:00[2025-08-20 10:32:32.526180] 

Downloading langchain_experimental-0.3.4-py3-none-any.whl (209 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 209.2/209.2 kB 197.6 MB/s eta 0:00:00[2025-08-20 10:32:32.540039] 

Downloading langchain_google_genai-2.1.5-py3-none-any.whl (44 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 44.8/44.8 kB 130.1 MB/s eta 0:00:00[2025-08-20 10:32:32.552816] 

Downloading langchain_google_vertexai-2.0.27-py3-none-any.whl (101 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 101.0/101.0 kB 121.7 MB/s eta 0:00:00[2025-08-20 10:32:32.568426] 

Downloading langchain_groq-0.3.2-py3-none-any.whl (15 kB)

Downloading langchain_text_splitters-0.3.8-py3-none-any.whl (32 kB)

Downloading langsmith-0.3.45-py3-none-any.whl (363 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 363.0/363.0 kB 123.0 MB/s eta 0:00:00[2025-08-20 10:32:32.605534] 

Downloading mando-0.7.1-py2.py3-none-any.whl (28 kB)

Downloading markdown_it_py-3.0.0-py3-none-any.whl (87 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 87.5/87.5 kB 126.9 MB/s eta 0:00:00[2025-08-20 10:32:32.630999] 

Downloading MarkupSafe-3.0.2-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (23 kB)

Downloading marshmallow-3.26.1-py3-none-any.whl (50 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 50.9/50.9 kB 98.6 MB/s eta 0:00:00[2025-08-20 10:32:32.653159] 

Downloading matplotlib-3.10.3-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (8.6 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8.6/8.6 MB 194.1 MB/s eta 0:00:00[2025-08-20 10:32:32.711701] 

Downloading matplotlib_inline-0.1.7-py3-none-any.whl (9.9 kB)

Downloading mccabe-0.7.0-py2.py3-none-any.whl (7.3 kB)

Downloading mdurl-0.1.2-py3-none-any.whl (10.0 kB)

Downloading mistune-3.1.0-py3-none-any.whl (53 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 53.7/53.7 kB 148.4 MB/s eta 0:00:00[2025-08-20 10:32:32.749335] 

Downloading mmh3-5.1.0-cp313-cp313-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (101 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 101.6/101.6 kB 191.1 MB/s eta 0:00:00[2025-08-20 10:32:32.761635] 

Downloading mpmath-1.3.0-py3-none-any.whl (536 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 536.2/536.2 kB 221.8 MB/s eta 0:00:00[2025-08-20 10:32:32.776069] 

Downloading multidict-6.3.2-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (248 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 248.5/248.5 kB 189.4 MB/s eta 0:00:00[2025-08-20 10:32:32.790030] 

Downloading mypy_extensions-1.1.0-py3-none-any.whl (5.0 kB)

Downloading narwhals-1.43.1-py3-none-any.whl (362 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 362.7/362.7 kB 205.9 MB/s eta 0:00:00[2025-08-20 10:32:32.810522] 

Downloading nbclient-0.10.2-py3-none-any.whl (25 kB)

Downloading nbconvert-7.16.5-py3-none-any.whl (258 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 258.1/258.1 kB 130.0 MB/s eta 0:00:00[2025-08-20 10:32:32.836944] 

Downloading nbformat-5.10.4-py3-none-any.whl (78 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 78.5/78.5 kB 129.0 MB/s eta 0:00:00[2025-08-20 10:32:32.849500] 

Downloading nest_asyncio-1.6.0-py3-none-any.whl (5.2 kB)

Downloading networkx-3.5-py3-none-any.whl (2.0 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.0/2.0 MB 136.5 MB/s eta 0:00:00[2025-08-20 10:32:32.886953] 

Downloading notebook-7.3.2-py3-none-any.whl (13.2 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 13.2/13.2 MB 137.0 MB/s eta 0:00:00[2025-08-20 10:32:33.003555] 

Downloading notebook_shim-0.2.4-py3-none-any.whl (13 kB)

Downloading numexpr-2.11.0-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (406 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 406.6/406.6 kB 144.4 MB/s eta 0:00:00[2025-08-20 10:32:33.031111] 

Downloading numpy-2.2.2-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (16.1 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 16.1/16.1 MB 141.4 MB/s eta 0:00:00[2025-08-20 10:32:33.162803] 

Downloading oauthlib-3.3.1-py3-none-any.whl (160 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 160.1/160.1 kB 139.4 MB/s eta 0:00:00[2025-08-20 10:32:33.176806] 

Downloading onnxruntime-1.22.1-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (16.5 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 16.5/16.5 MB 181.7 MB/s eta 0:00:00[2025-08-20 10:32:33.280786] 

Downloading openai-0.28.0-py3-none-any.whl (76 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 76.5/76.5 kB 198.2 MB/s eta 0:00:00[2025-08-20 10:32:33.292988] 

Downloading openpyxl-3.1.5-py2.py3-none-any.whl (250 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 250.9/250.9 kB 218.9 MB/s eta 0:00:00[2025-08-20 10:32:33.305079] 

Downloading opentelemetry_api-1.35.0-py3-none-any.whl (65 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 65.6/65.6 kB 130.9 MB/s eta 0:00:00[2025-08-20 10:32:33.317945] 

Downloading opentelemetry_exporter_otlp_proto_common-1.35.0-py3-none-any.whl (18 kB)

Downloading opentelemetry_exporter_otlp_proto_grpc-1.35.0-py3-none-any.whl (18 kB)

Downloading opentelemetry_proto-1.35.0-py3-none-any.whl (72 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 72.5/72.5 kB 115.5 MB/s eta 0:00:00[2025-08-20 10:32:33.353248] 

Downloading opentelemetry_sdk-1.35.0-py3-none-any.whl (119 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 119.4/119.4 kB 122.2 MB/s eta 0:00:00[2025-08-20 10:32:33.367932] 

Downloading opentelemetry_semantic_conventions-0.56b0-py3-none-any.whl (201 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 201.6/201.6 kB 140.7 MB/s eta 0:00:00[2025-08-20 10:32:33.385483] 

Downloading oracledb-3.1.0-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.manylinux_2_28_x86_64.whl (2.8 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.8/2.8 MB 130.8 MB/s eta 0:00:00[2025-08-20 10:32:33.424385] 

Downloading orjson-3.10.18-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (133 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 133.1/133.1 kB 138.1 MB/s eta 0:00:00[2025-08-20 10:32:33.439794] 

Downloading overrides-7.7.0-py3-none-any.whl (17 kB)

Downloading packaging-24.2-py3-none-any.whl (65 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 65.5/65.5 kB 137.5 MB/s eta 0:00:00[2025-08-20 10:32:33.465182] 

Downloading pandas-2.3.1-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (12.1 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 12.1/12.1 MB 117.8 MB/s eta 0:00:00[2025-08-20 10:32:33.603796] 

Downloading pandas_gbq-0.29.2-py3-none-any.whl (40 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 41.0/41.0 kB 127.0 MB/s eta 0:00:00[2025-08-20 10:32:33.616705] 

Downloading pandocfilters-1.5.1-py2.py3-none-any.whl (8.7 kB)

Downloading parso-0.8.4-py2.py3-none-any.whl (103 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 103.7/103.7 kB 119.8 MB/s eta 0:00:00[2025-08-20 10:32:33.644145] 

Downloading pathspec-0.12.1-py3-none-any.whl (31 kB)

Downloading pillow-11.2.1-cp313-cp313-manylinux_2_28_x86_64.whl (4.6 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.6/4.6 MB 73.6 MB/s eta 0:00:00[2025-08-20 10:32:33.744261] 

Downloading pinecone_client-6.0.0-py3-none-any.whl (6.7 kB)

Downloading pinecone_plugin_interface-0.0.7-py3-none-any.whl (6.2 kB)

Downloading platformdirs-4.3.6-py3-none-any.whl (18 kB)

Downloading plotly-6.2.0-py3-none-any.whl (9.6 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 9.6/9.6 MB 118.6 MB/s eta 0:00:00[2025-08-20 10:32:33.882483] 

Downloading posthog-5.4.0-py3-none-any.whl (105 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 105.4/105.4 kB 109.0 MB/s eta 0:00:00[2025-08-20 10:32:33.896346] 

Downloading prometheus_client-0.21.1-py3-none-any.whl (54 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 54.7/54.7 kB 102.6 MB/s eta 0:00:00[2025-08-20 10:32:33.909446] 

Downloading prompt_toolkit-3.0.50-py3-none-any.whl (387 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 387.8/387.8 kB 137.3 MB/s eta 0:00:00[2025-08-20 10:32:33.925461] 

Downloading propcache-0.3.1-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (228 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 228.2/228.2 kB 152.6 MB/s eta 0:00:00[2025-08-20 10:32:33.941801] 

Downloading proto_plus-1.26.1-py3-none-any.whl (50 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 50.2/50.2 kB 119.0 MB/s eta 0:00:00[2025-08-20 10:32:33.955198] 

Downloading protobuf-6.31.1-cp39-abi3-manylinux2014_x86_64.whl (321 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 321.1/321.1 kB 130.9 MB/s eta 0:00:00[2025-08-20 10:32:33.972701] 

Downloading psutil-6.1.1-cp36-abi3-manylinux_2_12_x86_64.manylinux2010_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (287 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 287.5/287.5 kB 130.1 MB/s eta 0:00:00

Downloading pure_eval-0.2.3-py3-none-any.whl (11 kB)

Downloading pyarrow-19.0.1-cp313-cp313-manylinux_2_28_x86_64.whl (42.1 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 42.1/42.1 MB 183.4 MB/s eta 0:00:00[2025-08-20 10:32:34.292666] 

Downloading pyasn1-0.6.1-py3-none-any.whl (83 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 83.1/83.1 kB 197.7 MB/s eta 0:00:00[2025-08-20 10:32:34.303864] 

Downloading pyasn1_modules-0.4.2-py3-none-any.whl (181 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 181.3/181.3 kB 207.0 MB/s eta 0:00:00[2025-08-20 10:32:34.315538] 

Downloading pybase64-1.4.1-cp313-cp313-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (71 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 71.4/71.4 kB 205.5 MB/s eta 0:00:00[2025-08-20 10:32:34.328654] 

Downloading pycparser-2.22-py3-none-any.whl (117 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 117.6/117.6 kB 139.4 MB/s eta 0:00:00[2025-08-20 10:32:34.340500] 

Downloading pydantic-2.11.2-py3-none-any.whl (443 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 443.3/443.3 kB 141.6 MB/s eta 0:00:00[2025-08-20 10:32:34.357366] 

Downloading pydantic_settings-2.10.0-py3-none-any.whl (45 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 45.2/45.2 kB 121.8 MB/s eta 0:00:00[2025-08-20 10:32:34.371476] 

Downloading pydantic_core-2.33.1-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (2.0 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.0/2.0 MB 132.5 MB/s eta 0:00:00[2025-08-20 10:32:34.406815] 

Downloading pydata_google_auth-1.9.1-py2.py3-none-any.whl (15 kB)

Downloading pydeck-0.9.1-py2.py3-none-any.whl (6.9 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 6.9/6.9 MB 127.6 MB/s eta 0:00:00[2025-08-20 10:32:34.484426] 

Downloading pygments-2.19.1-py3-none-any.whl (1.2 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.2/1.2 MB 148.3 MB/s eta 0:00:00[2025-08-20 10:32:34.503467] 

Downloading pylint-3.3.7-py3-none-any.whl (522 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 522.6/522.6 kB 128.5 MB/s eta 0:00:00[2025-08-20 10:32:34.521176] 

Downloading pyparsing-3.2.3-py3-none-any.whl (111 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 111.1/111.1 kB 127.2 MB/s eta 0:00:00[2025-08-20 10:32:34.534590] 

Downloading pypdf-5.8.0-py3-none-any.whl (309 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 309.7/309.7 kB 131.9 MB/s eta 0:00:00[2025-08-20 10:32:34.550255] 

Downloading pyproject_hooks-1.2.0-py3-none-any.whl (10 kB)

Downloading pyreadline3-3.5.4-py3-none-any.whl (83 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 83.2/83.2 kB 105.8 MB/s eta 0:00:00[2025-08-20 10:32:34.572510] 

Downloading python_dateutil-2.9.0.post0-py2.py3-none-any.whl (229 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 229.9/229.9 kB 129.4 MB/s eta 0:00:00[2025-08-20 10:32:34.586814] 

Downloading python_dotenv-1.1.0-py3-none-any.whl (20 kB)

Downloading python_json_logger-3.2.1-py3-none-any.whl (14 kB)

Downloading pytz-2024.2-py2.py3-none-any.whl (508 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 508.0/508.0 kB 224.7 MB/s eta 0:00:00[2025-08-20 10:32:34.617535] 

Downloading PyYAML-6.0.2-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (759 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 759.5/759.5 kB 223.0 MB/s eta 0:00:00[2025-08-20 10:32:34.632132] 

Downloading pyzmq-26.2.0-cp313-cp313-manylinux_2_28_x86_64.whl (860 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 860.3/860.3 kB 193.4 MB/s eta 0:00:00[2025-08-20 10:32:34.650968] 

Downloading radon-6.0.1-py2.py3-none-any.whl (52 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 52.8/52.8 kB 186.5 MB/s eta 0:00:00[2025-08-20 10:32:34.663263] 

Downloading referencing-0.36.1-py3-none-any.whl (26 kB)

Downloading regex-2024.11.6-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (796 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 796.9/796.9 kB 207.3 MB/s eta 0:00:00[2025-08-20 10:32:34.691708] 

Downloading requests-2.32.4-py3-none-any.whl (64 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 64.8/64.8 kB 148.4 MB/s eta 0:00:00[2025-08-20 10:32:34.704230] 

Downloading requests_oauthlib-2.0.0-py2.py3-none-any.whl (24 kB)

Downloading requests_toolbelt-1.0.0-py2.py3-none-any.whl (54 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 54.5/54.5 kB 100.7 MB/s eta 0:00:00[2025-08-20 10:32:34.727765] 

Downloading rfc3339_validator-0.1.4-py2.py3-none-any.whl (3.5 kB)

Downloading rfc3986_validator-0.1.1-py2.py3-none-any.whl (4.2 kB)

Downloading rich-14.0.0-py3-none-any.whl (243 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 243.2/243.2 kB 132.0 MB/s eta 0:00:00[2025-08-20 10:32:34.760783] 

Downloading rpds_py-0.22.3-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (385 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 385.2/385.2 kB 131.4 MB/s eta 0:00:00[2025-08-20 10:32:34.776981] 

Downloading rsa-4.9.1-py3-none-any.whl (34 kB)

Downloading safetensors-0.5.3-cp38-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (471 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 471.6/471.6 kB 144.5 MB/s eta 0:00:00[2025-08-20 10:32:34.803066] 

Downloading scikit_learn-1.7.1-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (9.4 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 9.4/9.4 MB 116.5 MB/s eta 0:00:00[2025-08-20 10:32:34.901200] 

Downloading scipy-1.16.0-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (35.1 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 35.1/35.1 MB 208.8 MB/s eta 0:00:00[2025-08-20 10:32:35.145981] 

Downloading seaborn-0.13.2-py3-none-any.whl (294 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 294.9/294.9 kB 224.4 MB/s eta 0:00:00[2025-08-20 10:32:35.158299] 

Downloading Send2Trash-1.8.3-py3-none-any.whl (18 kB)

Downloading sentence_transformers-5.0.0-py3-none-any.whl (470 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 470.2/470.2 kB 167.6 MB/s eta 0:00:00[2025-08-20 10:32:35.182641] 

Downloading setuptools-75.8.0-py3-none-any.whl (1.2 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.2/1.2 MB 156.4 MB/s eta 0:00:00[2025-08-20 10:32:35.203791] 

Downloading shapely-2.1.1-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (3.1 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.1/3.1 MB 199.0 MB/s eta 0:00:00[2025-08-20 10:32:35.235485] 

Downloading shellingham-1.5.4-py2.py3-none-any.whl (9.8 kB)

Downloading six-1.17.0-py2.py3-none-any.whl (11 kB)

Downloading smmap-5.0.2-py3-none-any.whl (24 kB)

Downloading sniffio-1.3.1-py3-none-any.whl (10 kB)

Downloading soupsieve-2.6-py3-none-any.whl (36 kB)

Downloading sqlalchemy-2.0.41-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (3.2 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.2/3.2 MB 189.0 MB/s eta 0:00:00[2025-08-20 10:32:35.307051] 

Downloading sqlparse-0.5.3-py3-none-any.whl (44 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 44.4/44.4 kB 204.2 MB/s eta 0:00:00[2025-08-20 10:32:35.317465] 

Downloading stack_data-0.6.3-py3-none-any.whl (24 kB)

Downloading streamlit-1.46.0-py3-none-any.whl (10.1 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 10.1/10.1 MB 214.0 MB/s eta 0:00:00[2025-08-20 10:32:35.389777] 

Downloading sympy-1.14.0-py3-none-any.whl (6.3 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 6.3/6.3 MB 215.5 MB/s eta 0:00:00[2025-08-20 10:32:35.430364] 

Downloading tabulate-0.9.0-py3-none-any.whl (35 kB)

Downloading tenacity-8.5.0-py3-none-any.whl (28 kB)

Downloading terminado-0.18.1-py3-none-any.whl (14 kB)

Downloading threadpoolctl-3.6.0-py3-none-any.whl (18 kB)

Downloading tinycss2-1.4.0-py3-none-any.whl (26 kB)

Downloading tokenizers-0.21.2-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (3.1 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.1/3.1 MB 195.4 MB/s eta 0:00:00[2025-08-20 10:32:35.504618] 

Downloading toml-0.10.2-py2.py3-none-any.whl (16 kB)

Downloading tomlkit-0.13.3-py3-none-any.whl (38 kB)

Downloading torch-2.7.1-cp313-cp313-manylinux_2_28_x86_64.whl (821.0 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 821.0/821.0 MB 176.5 MB/s eta 0:00:00[2025-08-20 10:32:41.181137] 

Downloading tornado-6.4.2-cp38-abi3-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (437 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 437.2/437.2 kB 108.0 MB/s eta 0:00:00[2025-08-20 10:32:41.197580] 

Downloading tqdm-4.67.1-py3-none-any.whl (78 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 78.5/78.5 kB 118.5 MB/s eta 0:00:00[2025-08-20 10:32:41.210472] 

Downloading traitlets-5.14.3-py3-none-any.whl (85 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 85.4/85.4 kB 117.4 MB/s eta 0:00:00[2025-08-20 10:32:41.223547] 

Downloading transformers-4.53.2-py3-none-any.whl (10.8 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 10.8/10.8 MB 106.3 MB/s eta 0:00:00[2025-08-20 10:32:41.341646] 

Downloading typer-0.16.0-py3-none-any.whl (46 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 46.3/46.3 kB 90.1 MB/s eta 0:00:00[2025-08-20 10:32:41.354251] 

Downloading types_python_dateutil-2.9.0.20241206-py3-none-any.whl (14 kB)

Downloading typing_inspect-0.9.0-py3-none-any.whl (8.8 kB)

Downloading typing_inspection-0.4.0-py3-none-any.whl (14 kB)

Downloading typing_extensions-4.12.2-py3-none-any.whl (37 kB)

Downloading tzdata-2025.1-py2.py3-none-any.whl (346 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 346.8/346.8 kB 112.8 MB/s eta 0:00:00[2025-08-20 10:32:41.408196] 

Downloading uri_template-1.3.0-py3-none-any.whl (11 kB)

Downloading urllib3-2.5.0-py3-none-any.whl (129 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 129.8/129.8 kB 124.5 MB/s eta 0:00:00[2025-08-20 10:32:41.432516] 

Downloading uvicorn-0.35.0-py3-none-any.whl (66 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 66.4/66.4 kB 116.8 MB/s eta 0:00:00[2025-08-20 10:32:41.446866] 

Downloading validators-0.35.0-py3-none-any.whl (44 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 44.7/44.7 kB 124.9 MB/s eta 0:00:00[2025-08-20 10:32:41.459404] 

Downloading watchdog-6.0.0-py3-none-manylinux2014_x86_64.whl (79 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 79.1/79.1 kB 127.3 MB/s eta 0:00:00[2025-08-20 10:32:41.471923] 

Downloading watchfiles-1.1.0-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (451 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 451.7/451.7 kB 228.6 MB/s eta 0:00:00[2025-08-20 10:32:41.488606] 

Downloading wcwidth-0.2.13-py2.py3-none-any.whl (34 kB)

Downloading webcolors-24.11.1-py3-none-any.whl (14 kB)

Downloading webencodings-0.5.1-py2.py3-none-any.whl (11 kB)

Downloading websocket_client-1.8.0-py3-none-any.whl (58 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 58.8/58.8 kB 144.5 MB/s eta 0:00:00[2025-08-20 10:32:41.524480] 

Downloading websockets-15.0.1-cp313-cp313-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (182 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 182.5/182.5 kB 147.2 MB/s eta 0:00:00[2025-08-20 10:32:41.537860] 

Downloading werkzeug-3.1.3-py3-none-any.whl (224 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 224.5/224.5 kB 156.9 MB/s eta 0:00:00[2025-08-20 10:32:41.550473] 

Downloading widgetsnbextension-4.0.13-py3-none-any.whl (2.3 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.3/2.3 MB 176.2 MB/s eta 0:00:00[2025-08-20 10:32:41.579399] 

Downloading yarl-1.19.0-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (347 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 347.6/347.6 kB 181.8 MB/s eta 0:00:00[2025-08-20 10:32:41.594892] 

Downloading zipp-3.23.0-py3-none-any.whl (10 kB)

Downloading zstandard-0.23.0-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (5.4 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5.4/5.4 MB 82.0 MB/s eta 0:00:00[2025-08-20 10:32:41.684030] 

Downloading nvidia_cublas_cu12-12.6.4.1-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (393.1 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 393.1/393.1 MB 121.0 MB/s eta 0:00:00[2025-08-20 10:32:44.553312] 

Downloading nvidia_cuda_cupti_cu12-12.6.80-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (8.9 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8.9/8.9 MB 120.0 MB/s eta 0:00:00[2025-08-20 10:32:44.640965] 

Downloading nvidia_cuda_nvrtc_cu12-12.6.77-py3-none-manylinux2014_x86_64.whl (23.7 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 23.7/23.7 MB 134.8 MB/s eta 0:00:00[2025-08-20 10:32:44.844347] 

Downloading nvidia_cuda_runtime_cu12-12.6.77-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (897 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 897.7/897.7 kB 179.8 MB/s eta 0:00:00[2025-08-20 10:32:44.861921] 

Downloading nvidia_cudnn_cu12-9.5.1.17-py3-none-manylinux_2_28_x86_64.whl (571.0 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 571.0/571.0 MB 125.5 MB/s eta 0:00:00[2025-08-20 10:32:48.651637] 

Downloading nvidia_cufft_cu12-11.3.0.4-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (200.2 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 200.2/200.2 MB 179.4 MB/s eta 0:00:00[2025-08-20 10:32:50.150992] 

Downloading nvidia_cufile_cu12-1.11.1.6-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (1.1 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.1/1.1 MB 153.6 MB/s eta 0:00:00[2025-08-20 10:32:50.171504] 

Downloading nvidia_curand_cu12-10.3.7.77-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (56.3 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 56.3/56.3 MB 199.0 MB/s eta 0:00:00[2025-08-20 10:32:50.525408] 

Downloading nvidia_cusolver_cu12-11.7.1.2-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (158.2 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 158.2/158.2 MB 111.9 MB/s eta 0:00:00[2025-08-20 10:32:51.688545] 

Downloading nvidia_cusparse_cu12-12.5.4.2-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (216.6 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 216.6/216.6 MB 149.3 MB/s eta 0:00:00[2025-08-20 10:32:53.143995] 

Downloading nvidia_cusparselt_cu12-0.6.3-py3-none-manylinux2014_x86_64.whl (156.8 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 156.8/156.8 MB 184.6 MB/s eta 0:00:00[2025-08-20 10:32:54.410158] 

Downloading nvidia_nccl_cu12-2.26.2-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (201.3 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 201.3/201.3 MB 126.1 MB/s eta 0:00:00[2025-08-20 10:32:55.831650] 

Downloading nvidia_nvjitlink_cu12-12.6.85-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl (19.7 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 19.7/19.7 MB 175.8 MB/s eta 0:00:00[2025-08-20 10:32:56.000233] 

Downloading nvidia_nvtx_cu12-12.6.77-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (89 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 89.3/89.3 kB 117.2 MB/s eta 0:00:00[2025-08-20 10:32:56.012468] 

Downloading triton-3.3.1-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (155.7 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 155.7/155.7 MB 88.6 MB/s eta 0:00:00[2025-08-20 10:32:57.234834] 

Downloading hf_xet-1.1.8-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (3.2 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.2/3.2 MB 169.0 MB/s eta 0:00:00

Downloading pexpect-4.9.0-py2.py3-none-any.whl (63 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 63.8/63.8 kB 185.9 MB/s eta 0:00:00[2025-08-20 10:32:57.283707] 

Downloading ptyprocess-0.7.0-py2.py3-none-any.whl (13 kB)

Downloading uvloop-0.21.0-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (4.7 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.7/4.7 MB 163.1 MB/s eta 0:00:00[2025-08-20 10:32:57.337223] 

Building wheels for collected packages: PyPika, pywinpty, sqlalchemy_bigquery

  Building wheel for PyPika (pyproject.toml): started

  Building wheel for PyPika (pyproject.toml): finished with status 'done'

  Created wheel for PyPika: filename=pypika-0.48.9-py2.py3-none-any.whl size=53803 sha256=2b365b3575709056ce6e49104653d6fdd755366751a81de6937022b37c52f1e6

  Stored in directory: /tmp/pip-ephem-wheel-cache-_om7rxb1/wheels/b4/f8/a5/28e9c1524d320f4b8eefdce0e487b5c2e128dbf2ed1bb4a60b

  Building wheel for pywinpty (pyproject.toml): started

  Building wheel for pywinpty (pyproject.toml): finished with status 'error'

  error: subprocess-exited-with-error

  

  × Building wheel for pywinpty (pyproject.toml) did not run successfully.

  │ exit code: 1

  ╰─> [119 lines of output]

      Running `maturin pep517 build-wheel -i /home/adminuser/venv/bin/python --compatibility off`

      Python reports SOABI: cpython-313-x86_64-linux-gnu

      Computed rustc target triple: x86_64-unknown-linux-gnu

      Installation directory: /home/adminuser/.cache/puccinialin

      Rustup already downloaded

      Installing rust to /home/adminuser/.cache/puccinialin/rustup

      warn: It looks like you have an existing rustup settings file at:

      warn: /home/adminuser/.rustup/settings.toml

      warn: Rustup will install the default toolchain as specified in the settings file,

      warn: instead of the one inferred from the default host triple.

      info: profile set to 'minimal'

      info: default host triple is x86_64-unknown-linux-gnu

      warn: Updating existing toolchain, profile choice will be ignored

      info: syncing channel updates for 'stable-x86_64-unknown-linux-gnu'

      info: default toolchain set to 'stable-x86_64-unknown-linux-gnu'

      Checking if cargo is installed

      cargo 1.89.0 (c24e10642 2025-06-23)

      ⚠️  Warning: `project.version` field is required in pyproject.toml unless it is present in the `project.dynamic` list

      📦 Including license file `LICENSE.txt`

      🍹 Building a mixed python/rust project

      🔗 Found pyo3 bindings

      🐍 Found CPython 3.13 at /home/adminuser/venv/bin/python

         Compiling proc-macro2 v1.0.88

         Compiling windows_x86_64_gnu v0.52.6

         Compiling unicode-ident v1.0.13

         Compiling target-lexicon v0.12.16

         Compiling autocfg v1.4.0

         Compiling once_cell v1.20.2

         Compiling rustix v0.38.37

         Compiling bitflags v2.6.0

         Compiling linux-raw-sys v0.4.14

         Compiling libc v0.2.160

         Compiling either v1.13.0

         Compiling home v0.5.9

         Compiling heck v0.5.0

         Compiling cfg-if v1.0.0

         Compiling unindent v0.2.3

         Compiling indoc v2.0.5

         Compiling windows-targets v0.52.6

         Compiling num-traits v0.2.19

         Compiling memoffset v0.9.1

         Compiling windows-result v0.2.0

         Compiling windows-strings v0.1.0

         Compiling pyo3-build-config v0.22.5

         Compiling quote v1.0.37

         Compiling syn v2.0.79

         Compiling pyo3-macros-backend v0.22.5

         Compiling pyo3-ffi v0.22.5

         Compiling pyo3 v0.22.5

         Compiling which v6.0.3

         Compiling windows-implement v0.58.0

         Compiling windows-interface v0.58.0

         Compiling enum-primitive-derive v0.3.0

         Compiling windows-core v0.58.0

         Compiling windows v0.58.0

         Compiling winpty-rs v0.4.0

      error[E0433]: failed to resolve: could not find `Win32` in `windows`

       --> /home/adminuser/.cache/puccinialin/cargo/registry/src/index.crates.io-1949cf8c6b5b557f/winpty-rs-0.4.0/src/pty/base.rs:3:14

        |

      3 | use windows::Win32::Foundation::{CloseHandle, HANDLE, STATUS_PENDING, S_OK, WAIT_FAILED, WAIT_OBJECT_0, WAIT_TIMEOUT};

        |              ^^^^^ could not find `Win32` in `windows`

      

      error[E0433]: failed to resolve: could not find `Win32` in `windows`

       --> /home/adminuser/.cache/puccinialin/cargo/registry/src/index.crates.io-1949cf8c6b5b557f/winpty-rs-0.4.0/src/pty/base.rs:4:14

        |

      4 | use windows::Win32::Storage::FileSystem::{GetFileSizeEx, ReadFile, WriteFile};

        |              ^^^^^ could not find `Win32` in `windows`

      

      error[E0433]: failed to resolve: could not find `Win32` in `windows`

       --> /home/adminuser/.cache/puccinialin/cargo/registry/src/index.crates.io-1949cf8c6b5b557f/winpty-rs-0.4.0/src/pty/base.rs:5:14

        |

      5 | use windows::Win32::System::Pipes::PeekNamedPipe;

        |              ^^^^^ could not find `Win32` in `windows`

      

      error[E0433]: failed to resolve: could not find `Win32` in `windows`

       --> /home/adminuser/.cache/puccinialin/cargo/registry/src/index.crates.io-1949cf8c6b5b557f/winpty-rs-0.4.0/src/pty/base.rs:6:14

        |

      6 | use windows::Win32::System::IO::CancelIoEx;

        |              ^^^^^ could not find `Win32` in `windows`

      

      error[E0433]: failed to resolve: could not find `Win32` in `windows`

       --> /home/adminuser/.cache/puccinialin/cargo/registry/src/index.crates.io-1949cf8c6b5b557f/winpty-rs-0.4.0/src/pty/base.rs:7:14

        |

      7 | use windows::Win32::System::Threading::{GetExitCodeProcess, GetProcessId, WaitForSingleObject};

        |              ^^^^^ could not find `Win32` in `windows`

      

      error[E0433]: failed to resolve: could not find `Win32` in `windows`

       --> /home/adminuser/.cache/puccinialin/cargo/registry/src/index.crates.io-1949cf8c6b5b557f/winpty-rs-0.4.0/src/pty/base.rs:8:14

        |

      8 | use windows::Win32::Globalization::{MultiByteToWideChar, WideCharToMultiByte, CP_UTF8, MULTI_BYTE_TO_WIDE_CHAR_FLAGS};

        |              ^^^^^ could not find `Win32` in `windows`

      

      error[E0433]: failed to resolve: could not find `Win32` in `windows`

        --> /home/adminuser/.cache/puccinialin/cargo/registry/src/index.crates.io-1949cf8c6b5b557f/winpty-rs-0.4.0/src/pty/base.rs:10:14

         |

      10 | use windows::Win32::System::Threading::INFINITE;

         |              ^^^^^ could not find `Win32` in `windows`

      

      error[E0432]: unresolved import `windows::core`

       --> /home/adminuser/.cache/puccinialin/cargo/registry/src/index.crates.io-1949cf8c6b5b557f/winpty-rs-0.4.0/src/pty/base.rs:9:14

        |

      9 | use windows::core::{HRESULT, Error, PCSTR};

        |              ^^^^ could not find `core` in `windows`

      

      error: the `Self` constructor can only be used with tuple or unit structs

        --> /home/adminuser/.cache/puccinialin/cargo/registry/src/index.crates.io-1949cf8c6b5b557f/winpty-rs-0.4.0/src/pty/base.rs:70:9

         |

      70 |         Self(value.0)

         |         ^^^^^^^^^^^^^

      

      Some errors have detailed explanations: E0432, E0433.

      For more information about an error, try `rustc --explain E0432`.

      error: could not compile `winpty-rs` (lib) due to 9 previous errors

      warning: build failed, waiting for other jobs to finish...

      💥 maturin failed

        Caused by: Failed to build a native library through cargo

        Caused by: Cargo build finished with "exit status: 101": `env -u CARGO PYO3_ENVIRONMENT_SIGNATURE="cpython-3.13-64bit" PYO3_PYTHON="/home/adminuser/venv/bin/python" PYTHON_SYS_EXECUTABLE="/home/adminuser/venv/bin/python" "cargo" "rustc" "--message-format" "json-render-diagnostics" "--manifest-path" "/tmp/pip-install-ic3engfr/pywinpty_899eef6758444f8abe67174c4b01e877/Cargo.toml" "--release" "--lib"`

      Rust not found, installing into a temporary directory

      Error: command ['maturin', 'pep517', 'build-wheel', '-i', '/home/adminuser/venv/bin/python', '--compatibility', 'off'] returned non-zero exit status 1

      [end of output]

  

  note: This error originates from a subprocess, and is likely not a problem with pip.

  ERROR: Failed building wheel for pywinpty

  Building wheel for sqlalchemy_bigquery (pyproject.toml): started

  Building wheel for sqlalchemy_bigquery (pyproject.toml): finished with status 'done'

  Created wheel for sqlalchemy_bigquery: filename=sqlalchemy_bigquery-0.0.7-py3-none-any.whl size=4620 sha256=efcb271b33a86c67ccbf7387a3ffe8affc831dd95f42f958098043e7bc47a84d

  Stored in directory: /tmp/pip-ephem-wheel-cache-_om7rxb1/wheels/18/f8/3a/e96ae18838495cc30ae741717b00fbcade0d081dd5d2a68dd2

Successfully built PyPika sqlalchemy_bigquery

Failed to build pywinpty

ERROR: Could not build wheels for pywinpty, which is required to install pyproject.toml-based projects


[notice] A new release of pip is available: 24.0 -> 25.2

[notice] To update, run: pip install --upgrade pip

Checking if Streamlit is installed

Installing rich for an improved exception logging

Using standard pip install.

Collecting rich>=10.14.0

  Downloading rich-14.1.0-py3-none-any.whl.metadata (18 kB)

Collecting markdown-it-py>=2.2.0 (from rich>=10.14.0)

  Downloading markdown_it_py-4.0.0-py3-none-any.whl.metadata (7.3 kB)

Collecting pygments<3.0.0,>=2.13.0 (from rich>=10.14.0)

  Downloading pygments-2.19.2-py3-none-any.whl.metadata (2.5 kB)

Collecting mdurl~=0.1 (from markdown-it-py>=2.2.0->rich>=10.14.0)

  Downloading mdurl-0.1.2-py3-none-any.whl.metadata (1.6 kB)

Downloading rich-14.1.0-py3-none-any.whl (243 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 243.4/243.4 kB 14.1 MB/s eta 0:00:00[2025-08-20 10:33:31.192341] 

Downloading markdown_it_py-4.0.0-py3-none-any.whl (87 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 87.3/87.3 kB 117.7 MB/s eta 0:00:00[2025-08-20 10:33:31.206824] 

Downloading pygments-2.19.2-py3-none-any.whl (1.2 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.2/1.2 MB 86.5 MB/s eta 0:00:00[2025-08-20 10:33:31.235324] 

Downloading mdurl-0.1.2-py3-none-any.whl (10.0 kB)

Installing collected packages: pygments, mdurl, markdown-it-py, rich

  Attempting uninstall: pygments

    Found existing installation: Pygments 2.19.2

    Uninstalling Pygments-2.19.2:

      Successfully uninstalled Pygments-2.19.2

  Attempting uninstall: mdurl

    Found existing installation: mdurl 0.1.2

    Uninstalling mdurl-0.1.2:

      Successfully uninstalled mdurl-0.1.2

  Attempting uninstall: markdown-it-py

    Found existing installation: markdown-it-py 4.0.0

    Uninstalling markdown-it-py-4.0.0:

      Successfully uninstalled markdown-it-py-4.0.0

  Attempting uninstall: rich

    Found existing installation: rich 14.1.0

    Uninstalling rich-14.1.0:

      Successfully uninstalled rich-14.1.0

Successfully installed markdown-it-py-4.0.0 mdurl-0.1.2 pygments-2.19.2 rich-14.1.0


[notice] A new release of pip is available: 24.0 -> 25.2

[notice] To update, run: pip install --upgrade pip


────────────────────────────────────────────────────────────────────────────────────────


[10:33:33] ❗️ installer returned a non-zero exit code

[10:33:33] ❗️ Error during processing dependencies! Please fix the error and push an update, or try restarting the app.

[10:30:32] 🚀 Starting up repository: 'chatbot-gaib', branch: 'main', main module: 'newwneww.py'

[10:30:32] 🐙 Cloning repository...

[10:30:33] 🐙 Cloning into '/mount/src/chatbot-gaib'...

[10:30:33] 🐙 Cloned repository!

[10:30:33] 🐙 Pulling code changes from Github...

[10:30:34] 📦 Processing dependencies...


──────────────────────────────────────── uv ───────────────────────────────────────────


Using uv pip install.

Using Python 3.13.5 environment at /home/adminuser/venv

Resolved [2025-08-20 10:30:38.486529] 287 packages[2025-08-20 10:30:38.486837]  [2025-08-20 10:30:38.487147] in 3.75s[2025-08-20 10:30:38.487476] 

  × Failed to download and build `pywinpty==2.0.14`

  ╰─▶ Build backend failed to build wheel through `build_wheel` (exit status:

      1)


      [stdout]

      Running `maturin pep517 build-wheel -i

      /home/adminuser/.cache/uv/builds-v0/.tmpP3IsCQ/bin/python

      --compatibility off`

      Rust not found, installing into a temporary directory

      cargo 1.89.0 (c24e10642 2025-06-23)


      [stderr]

      Python reports SOABI: cpython-313-x86_64-linux-gnu

      Computed rustc target triple: x86_64-unknown-linux-gnu

      Installation directory: /home/adminuser/.cache/puccinialin

      Downloading rustup-init from

      https://static.rust-lang.org/rustup/dist/x86_64-unknown-linux-gnu/rustup-init

Downloading rustup-init:   0%|          | 0.00/20.9M [00:00<?,

Downloading rustup-init:  44%|████▍     | 9.15M/20.9M [00:00<00:00,

Downloading rustup-init: 100%|██████████| 20.9M/20.9M

      [00:00<00:00, 109MB/s]

      Installing rust to /home/adminuser/.cache/puccinialin/rustup

      info: profile set to 'minimal'

      info: default host triple is x86_64-unknown-linux-gnu

      info: syncing channel updates for 'stable-x86_64-unknown-linux-gnu'

      info: latest update on 2025-08-07, rust version 1.89.0 (29483883e

      2025-08-04)

      info: downloading component 'cargo'

      info: downloading component 'rust-std'

      info: downloading component 'rustc'

      info: installing component 'cargo'

      info: installing component 'rust-std'

      info: installing component 'rustc'

      info: default toolchain set to 'stable-x86_64-unknown-linux-gnu'

      Checking if cargo is installed

          Updating crates.io index

       Downloading crates ...

        Downloaded enum-primitive-derive v0.3.0

        Downloaded errno v0.3.9

        Downloaded either v1.13.0

        Downloaded bitflags v2.6.0

        Downloaded winpty-rs v0.4.0

        Downloaded windows-implement v0.58.0

        Downloaded windows-strings v0.1.0

        Downloaded windows-core v0.58.0

        Downloaded unicode-ident v1.0.13

        Downloaded pyo3-macros-backend v0.22.5

        Downloaded pyo3-ffi v0.22.5

        Downloaded proc-macro2 v1.0.88

        Downloaded[2025-08-20 10:31:22.938654]  once_cell v1.20.2

        Downloaded which v6.0.3

        Downloaded portable-atomic v1.9.0

        Downloaded target-lexicon v0.12.16

        Downloaded quote v1.0.37

        Downloaded pyo3-build-config v0.22.5

        Downloaded num-traits v0.2.19

        Downloaded memoffset v0.9.1

        Downloaded syn v2.0.79

        Downloaded windows-result v0.2.0

        Downloaded windows-interface v0.58.0

        Downloaded home v0.5.9

        Downloaded heck v0.5.0

        Downloaded rustix v0.38.37

        Downloaded windows-targets v0.52.6

        Downloaded windows_x86_64_gnullvm v0.52.6

        Downloaded windows_i686_gnullvm v0.52.6

        Downloaded windows_aarch64_gnullvm v0.52.6

        Downloaded unindent v0.2.3

        Downloaded winsafe v0.0.19

        Downloaded pyo3-macros v0.22.5

        Downloaded indoc v2.0.5

        Downloaded autocfg v1.4.0

        Downloaded pyo3 v0.22.5

        Downloaded cfg-if v1.0.0

        Downloaded libc v0.2.160

        Downloaded windows_x86_64_msvc v0.52.6

        Downloaded windows_x86_64_gnu v0.52.6

        Downloaded windows_i686_gnu v0.52.6

        Downloaded windows_aarch64_msvc v0.52.6

        Downloaded windows_i686_msvc v0.52.6

        Downloaded linux-raw-sys v0.4.14

        Downloaded windows-sys v0.52.0

        Downloaded windows v0.58.0

      ⚠️  Warning: `project.version` field is required in pyproject.toml unless

      it is present in the `project.dynamic` list

      📦 Including license file `LICENSE.txt`

      🍹 Building a mixed python/rust project

      🔗 Found pyo3 bindings

      🐍 Found CPython 3.13 at

      /home/adminuser/.cache/uv/builds-v0/.tmpP3IsCQ/bin/python

         Compiling proc-macro2 v1.0.88

         Compiling windows_x86_64_gnu v0.52.6

         Compiling unicode-ident v1.0.13

         Compiling target-lexicon v0.12.16

         Compiling autocfg v1.4.0

         Compiling once_cell v1.20.2

         Compiling rustix v0.38.37

         Compiling linux-raw-sys v0.4.14

         Compiling bitflags v2.6.0

         Compiling libc v0.2.160

         Compiling home v0.5.9

         Compiling either v1.13.0

         Compiling heck v0.5.0

         Compiling indoc v2.0.5

         Compiling unindent v0.2.3

         Compiling cfg-if v1.0.0

         Compiling num-traits v0.2.19

         Compiling[2025-08-20 10:31:22.939006]  memoffset v0.9.1

         Compiling windows-targets v0.52.6

         Compiling windows-result v0.2.0

         Compiling windows-strings v0.1.0

         Compiling pyo3-build-config v0.22.5

         Compiling quote v1.0.37

         Compiling syn v2.0.79

         Compiling pyo3-macros-backend v0.22.5

         Compiling pyo3-ffi v0.22.5

         Compiling pyo3 v0.22.5

         Compiling which v6.0.3

         Compiling windows-interface v0.58.0

         Compiling windows-implement v0.58.0

         Compiling enum-primitive-derive v0.3.0

         Compiling windows-core v0.58.0

         Compiling windows v0.58.0

         Compiling winpty-rs v0.4.0

      error[E0433]: failed to resolve: could not find `Win32` in `windows`

       -->

      /home/adminuser/.cache/puccinialin/cargo/registry/src/index.crates.io-1949cf8c6b5b557f/winpty-rs-0.4.0/src/pty/base.rs:3:14

        |

      3 | use windows::Win32::Foundation::{CloseHandle, HANDLE,

      STATUS_PENDING, S_OK, WAIT_FAILED, WAIT_OBJECT_0, WAIT_TIMEOUT};

        |              ^^^^^ could not find `Win32` in `windows`


      error[E0433]: failed to resolve: could not find `Win32` in `windows`

       -->

      /home/adminuser/.cache/puccinialin/cargo/registry/src/index.crates.io-1949cf8c6b5b557f/winpty-rs-0.4.0/src/pty/base.rs:4:14

        |

      4 | use windows::Win32::Storage::FileSystem::{GetFileSizeEx, ReadFile,

      WriteFile};

        |              ^^^^^ could not find `Win32` in `windows`


      error[E0433]: failed to resolve: could not find `Win32` in `windows`

       -->

      /home/adminuser/.cache/puccinialin/cargo/registry/src/index.crates.io-1949cf8c6b5b557f/winpty-rs-0.4.0/src/pty/base.rs:5:14

        |

      5 | use windows::Win32::System::Pipes::PeekNamedPipe;

        |[2025-08-20 10:31:22.939232]               ^^^^^ could not find `Win32` in `windows`


      error[E0433]: failed to resolve: could not find `Win32` in `windows`

       -->

      /home/adminuser/.cache/puccinialin/cargo/registry/src/index.crates.io-1949cf8c6b5b557f/winpty-rs-0.4.0/src/pty/base.rs:6:14

        |

      6 | use windows::Win32::System::IO::CancelIoEx;

        |              ^^^^^ could not find `Win32` in `windows`


      error[E0433]: failed to resolve: could not find `Win32` in `windows`

       -->

      /home/adminuser/.cache/puccinialin/cargo/registry/src/index.crates.io-1949cf8c6b5b557f/winpty-rs-0.4.0/src/pty/base.rs:7:14

        |

      7 | use windows::Win32::System::Threading::{GetExitCodeProcess,

      GetProcessId, WaitForSingleObject};

        |              ^^^^^ could not find `Win32` in `windows`


      error[E0433]: failed to resolve: could not find `Win32` in `windows`

       -->

      /home/adminuser/.cache/puccinialin/cargo/registry/src/index.crates.io-1949cf8c6b5b557f/winpty-rs-0.4.0/src/pty/base.rs:8:14

        |

      8 |[2025-08-20 10:31:22.939447]  use windows::Win32::Globalization::{MultiByteToWideChar,

      WideCharToMultiByte, CP_UTF8, MULTI_BYTE_TO_WIDE_CHAR_FLAGS};

        |              ^^^^^ could not find `Win32` in `windows`


      error[E0433]: failed to resolve: could not find `Win32` in `windows`

        -->

      /home/adminuser/.cache/puccinialin/cargo/registry/src/index.crates.io-1949cf8c6b5b557f/winpty-rs-0.4.0/src/pty/base.rs:10:14

         |

      10 | use windows::Win32::System::Threading::INFINITE;

         |              ^^^^^ could not find `Win32` in `windows`


      error[E0432]: unresolved import `windows::core`

       -->

      /home/adminuser/.cache/puccinialin/cargo/registry/src/index.crates.io-1949cf8c6b5b557f/winpty-rs-0.4.0/src/pty/base.rs:9:14

        |

      9 | use windows::core::{HRESULT, Error, PCSTR};

        |              ^^^^ could not find `core` in `windows`


      error: the `Self` constructor can only be used with tuple or unit

      structs

        -->

      /home/adminuser/.cache/puccinialin/cargo/registry/src/index.crates.io-1949cf8c6b5b557f/winpty-rs-0.4.0/src/pty/base.rs:70:9

         |

      70 |         Self(value.0)

         |         ^^^^^^^^^^^^^


      Some errors have detailed explanations: E0432, E0433.

      For more information about an error, try `rustc --explain E0432`.

      error: could not compile `winpty-rs` (lib) due to 9 previous errors

      warning[2025-08-20 10:31:22.939675] : build failed, waiting for other jobs to finish...

      💥 maturin failed

        Caused by: Failed to build a native library through cargo

        Caused by: Cargo build finished with "exit status: 101":

      `env -u CARGO PYO3_ENVIRONMENT_SIGNATURE="cpython-3.13-64bit"

      PYO3_PYTHON="/home/adminuser/.cache/uv/builds-v0/.tmpP3IsCQ/bin/python"

      PYTHON_SYS_EXECUTABLE="/home/adminuser/.cache/uv/builds-v0/.tmpP3IsCQ/bin/python"

      "cargo" "rustc" "--message-format"

      "json-render-diagnostics" "--manifest-path"

      "/home/adminuser/.cache/uv/sdists-v6/pypi/pywinpty/2.0.14/kH1wVMrpfysfldyou9NvN/src/Cargo.toml"

      "--release" "--lib"`

      Error: command ['maturin', 'pep517', 'build-wheel', '-i',

      '/home/adminuser/.cache/uv/builds-v0/.tmpP3IsCQ/bin/python',

      '--compatibility', 'off'] returned non-zero exit status 1


Checking if Streamlit is installed

Installing rich for an improved exception logging

Using uv pip install.

Using Python 3.13.5 environment at /home/adminuser/venv

Resolved 4 packages in 62ms

Prepared 3 packages in 114ms

Installed 4 packages in 17ms

 + markdown-it-py==4.0.0

 + mdurl==0.1.2

 + pygments==2.19.2

 + rich==14.1.0


────────────────────────────────────────────────────────────────────────────────────────



──────────────────────────────────────── pip ───────────────────────────────────────────


Using standard pip install.

Collecting aiohappyeyeballs==2.6.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 1))

  Downloading aiohappyeyeballs-2.6.1-py3-none-any.whl.metadata (5.9 kB)

Collecting aiohttp==3.11.16 (from -r /mount/src/chatbot-gaib/requirements.txt (line 2))

  Downloading aiohttp-3.11.16-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (7.7 kB)

Collecting aiosignal==1.3.2 (from -r /mount/src/chatbot-gaib/requirements.txt (line 3))

  Downloading aiosignal-1.3.2-py2.py3-none-any.whl.metadata (3.8 kB)

Collecting altair==5.5.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 4))

  Downloading altair-5.5.0-py3-none-any.whl.metadata (11 kB)

Collecting annotated-types==0.7.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 5))

  Downloading annotated_types-0.7.0-py3-none-any.whl.metadata (15 kB)

Collecting anyio==4.8.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 6))

  Downloading anyio-4.8.0-py3-none-any.whl.metadata (4.6 kB)

Collecting argon2-cffi==23.1.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 7))

  Downloading argon2_cffi-23.1.0-py3-none-any.whl.metadata (5.2 kB)

Collecting argon2-cffi-bindings==21.2.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 8))

  Downloading argon2_cffi_bindings-21.2.0-cp36-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.7 kB)

Collecting arrow==1.3.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 9))

  Downloading arrow-1.3.0-py3-none-any.whl.metadata (7.5 kB)

Collecting astroid==3.3.10 (from -r /mount/src/chatbot-gaib/requirements.txt (line 10))

  Downloading astroid-3.3.10-py3-none-any.whl.metadata (4.4 kB)

Collecting asttokens==3.0.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 11))

  Downloading asttokens-3.0.0-py3-none-any.whl.metadata (4.7 kB)

Collecting async-lru==2.0.4 (from -r /mount/src/chatbot-gaib/requirements.txt (line 12))

  Downloading async_lru-2.0.4-py3-none-any.whl.metadata (4.5 kB)

Collecting attrs==24.3.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 13))

  Downloading attrs-24.3.0-py3-none-any.whl.metadata (11 kB)

Collecting babel==2.16.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 14))

  Downloading babel-2.16.0-py3-none-any.whl.metadata (1.5 kB)

Collecting backoff==2.2.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 15))

  Downloading backoff-2.2.1-py3-none-any.whl.metadata (14 kB)

Collecting bcrypt==4.3.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 16))

  Downloading bcrypt-4.3.0-cp39-abi3-manylinux_2_28_x86_64.whl.metadata (10 kB)

Collecting beautifulsoup4==4.12.3 (from -r /mount/src/chatbot-gaib/requirements.txt (line 17))

  Downloading beautifulsoup4-4.12.3-py3-none-any.whl.metadata (3.8 kB)

Collecting black==25.1.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 18))

  Downloading black-25.1.0-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.manylinux_2_28_x86_64.whl.metadata (81 kB)

     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 81.3/81.3 kB 11.1 MB/s eta 0:00:00[2025-08-20 10:31:27.577261] 

Collecting bleach==6.2.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 19))

  Downloading bleach-6.2.0-py3-none-any.whl.metadata (30 kB)

Collecting blinker==1.9.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 20))

  Downloading blinker-1.9.0-py3-none-any.whl.metadata (1.6 kB)

Collecting Bottleneck==1.5.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 21))

  Downloading bottleneck-1.5.0-cp313-cp313-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (8.1 kB)

Collecting build==1.2.2.post1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 22))

  Downloading build-1.2.2.post1-py3-none-any.whl.metadata (6.5 kB)

Collecting cachetools==5.5.2 (from -r /mount/src/chatbot-gaib/requirements.txt (line 23))

  Downloading cachetools-5.5.2-py3-none-any.whl.metadata (5.4 kB)

Collecting certifi==2025.6.15 (from -r /mount/src/chatbot-gaib/requirements.txt (line 24))

  Downloading certifi-2025.6.15-py3-none-any.whl.metadata (2.4 kB)

Collecting cffi==1.17.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 25))

  Downloading cffi-1.17.1-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (1.5 kB)

Collecting charset-normalizer==3.4.2 (from -r /mount/src/chatbot-gaib/requirements.txt (line 26))

  Downloading charset_normalizer-3.4.2-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (35 kB)

Collecting chromadb==1.0.15 (from -r /mount/src/chatbot-gaib/requirements.txt (line 27))

  Downloading chromadb-1.0.15-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (7.0 kB)

Collecting click==8.1.8 (from -r /mount/src/chatbot-gaib/requirements.txt (line 28))

  Downloading click-8.1.8-py3-none-any.whl.metadata (2.3 kB)

Collecting colorama==0.4.6 (from -r /mount/src/chatbot-gaib/requirements.txt (line 29))

  Downloading colorama-0.4.6-py2.py3-none-any.whl.metadata (17 kB)

Collecting coloredlogs==15.0.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 30))

  Downloading coloredlogs-15.0.1-py2.py3-none-any.whl.metadata (12 kB)

Collecting comm==0.2.2 (from -r /mount/src/chatbot-gaib/requirements.txt (line 31))

  Downloading comm-0.2.2-py3-none-any.whl.metadata (3.7 kB)

Collecting contourpy==1.3.2 (from -r /mount/src/chatbot-gaib/requirements.txt (line 32))

  Downloading contourpy-1.3.2-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (5.5 kB)

Collecting cryptography==44.0.2 (from -r /mount/src/chatbot-gaib/requirements.txt (line 33))

  Downloading cryptography-44.0.2-cp39-abi3-manylinux_2_28_x86_64.whl.metadata (5.7 kB)

Collecting cycler==0.12.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 34))

  Downloading cycler-0.12.1-py3-none-any.whl.metadata (3.8 kB)

Collecting dataclasses-json==0.6.7 (from -r /mount/src/chatbot-gaib/requirements.txt (line 35))

  Downloading dataclasses_json-0.6.7-py3-none-any.whl.metadata (25 kB)

Collecting db-dtypes==1.4.3 (from -r /mount/src/chatbot-gaib/requirements.txt (line 36))

  Downloading db_dtypes-1.4.3-py3-none-any.whl.metadata (3.0 kB)

Collecting debugpy==1.8.12 (from -r /mount/src/chatbot-gaib/requirements.txt (line 37))

  Downloading debugpy-1.8.12-cp313-cp313-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (1.3 kB)

Collecting decorator==5.1.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 38))

  Downloading decorator-5.1.1-py3-none-any.whl.metadata (4.0 kB)

Collecting defusedxml==0.7.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 39))

  Downloading defusedxml-0.7.1-py2.py3-none-any.whl.metadata (32 kB)

Collecting dill==0.4.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 40))

  Downloading dill-0.4.0-py3-none-any.whl.metadata (10 kB)

Collecting distro==1.9.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 41))

  Downloading distro-1.9.0-py3-none-any.whl.metadata (6.8 kB)

Collecting docstring_parser==0.16 (from -r /mount/src/chatbot-gaib/requirements.txt (line 42))

  Downloading docstring_parser-0.16-py3-none-any.whl.metadata (3.0 kB)

Collecting durationpy==0.10 (from -r /mount/src/chatbot-gaib/requirements.txt (line 43))

  Downloading durationpy-0.10-py3-none-any.whl.metadata (340 bytes)

Collecting et_xmlfile==2.0.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 44))

  Downloading et_xmlfile-2.0.0-py3-none-any.whl.metadata (2.7 kB)

Collecting executing==2.2.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 45))

  Downloading executing-2.2.0-py2.py3-none-any.whl.metadata (8.9 kB)

Collecting Faker==35.0.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 46))

  Downloading Faker-35.0.0-py3-none-any.whl.metadata (15 kB)

Collecting fastjsonschema==2.21.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 47))

  Downloading fastjsonschema-2.21.1-py3-none-any.whl.metadata (2.2 kB)

Collecting filelock==3.18.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 48))

  Downloading filelock-3.18.0-py3-none-any.whl.metadata (2.9 kB)

Collecting filetype==1.2.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 49))

  Downloading filetype-1.2.0-py2.py3-none-any.whl.metadata (6.5 kB)

Collecting Flask==3.1.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 50))

  Downloading flask-3.1.0-py3-none-any.whl.metadata (2.7 kB)

Collecting flatbuffers==25.2.10 (from -r /mount/src/chatbot-gaib/requirements.txt (line 51))

  Downloading flatbuffers-25.2.10-py2.py3-none-any.whl.metadata (875 bytes)

Collecting fonttools==4.59.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 52))

  Downloading fonttools-4.59.0-cp313-cp313-manylinux1_x86_64.manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_5_x86_64.whl.metadata (107 kB)

     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 107.9/107.9 kB 117.0 MB/s eta 0:00:00[2025-08-20 10:31:30.698210] 

Collecting fqdn==1.5.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 53))

  Downloading fqdn-1.5.1-py3-none-any.whl.metadata (1.4 kB)

Collecting frozenlist==1.5.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 54))

  Downloading frozenlist-1.5.0-cp313-cp313-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (13 kB)

Collecting fsspec==2025.7.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 55))

  Downloading fsspec-2025.7.0-py3-none-any.whl.metadata (12 kB)

Collecting gitdb==4.0.12 (from -r /mount/src/chatbot-gaib/requirements.txt (line 56))

  Downloading gitdb-4.0.12-py3-none-any.whl.metadata (1.2 kB)

Collecting GitPython==3.1.44 (from -r /mount/src/chatbot-gaib/requirements.txt (line 57))

  Downloading GitPython-3.1.44-py3-none-any.whl.metadata (13 kB)

Collecting google-ai-generativelanguage==0.6.18 (from -r /mount/src/chatbot-gaib/requirements.txt (line 58))

  Downloading google_ai_generativelanguage-0.6.18-py3-none-any.whl.metadata (9.8 kB)

Collecting google-api-core==2.25.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 59))

  Downloading google_api_core-2.25.1-py3-none-any.whl.metadata (3.0 kB)

Collecting google-auth==2.40.3 (from -r /mount/src/chatbot-gaib/requirements.txt (line 60))

  Downloading google_auth-2.40.3-py2.py3-none-any.whl.metadata (6.2 kB)

Collecting google-auth-oauthlib==1.2.2 (from -r /mount/src/chatbot-gaib/requirements.txt (line 61))

  Downloading google_auth_oauthlib-1.2.2-py3-none-any.whl.metadata (2.7 kB)

Collecting google-cloud-aiplatform==1.105.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 62))

  Downloading google_cloud_aiplatform-1.105.0-py2.py3-none-any.whl.metadata (38 kB)

Collecting google-cloud-bigquery==3.34.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 63))

  Downloading google_cloud_bigquery-3.34.0-py3-none-any.whl.metadata (8.0 kB)

Collecting google-cloud-bigquery-storage==2.32.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 64))

  Downloading google_cloud_bigquery_storage-2.32.0-py3-none-any.whl.metadata (9.8 kB)

Collecting google-cloud-core==2.4.3 (from -r /mount/src/chatbot-gaib/requirements.txt (line 65))

  Downloading google_cloud_core-2.4.3-py2.py3-none-any.whl.metadata (2.7 kB)

Collecting google-cloud-resource-manager==1.14.2 (from -r /mount/src/chatbot-gaib/requirements.txt (line 66))

  Downloading google_cloud_resource_manager-1.14.2-py3-none-any.whl.metadata (9.6 kB)

Collecting google-cloud-storage==2.19.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 67))

  Downloading google_cloud_storage-2.19.0-py2.py3-none-any.whl.metadata (9.1 kB)

Collecting google-crc32c==1.7.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 68))

  Downloading google_crc32c-1.7.1-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (2.3 kB)

Collecting google-genai==1.21.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 69))

  Downloading google_genai-1.21.1-py3-none-any.whl.metadata (37 kB)

Collecting google-resumable-media==2.7.2 (from -r /mount/src/chatbot-gaib/requirements.txt (line 70))

  Downloading google_resumable_media-2.7.2-py2.py3-none-any.whl.metadata (2.2 kB)

Collecting googleapis-common-protos==1.70.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 71))

  Downloading googleapis_common_protos-1.70.0-py3-none-any.whl.metadata (9.3 kB)

Collecting greenlet==3.2.3 (from -r /mount/src/chatbot-gaib/requirements.txt (line 72))

  Downloading greenlet-3.2.3-cp313-cp313-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl.metadata (4.1 kB)

Collecting groq==0.28.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 73))

  Downloading groq-0.28.0-py3-none-any.whl.metadata (15 kB)

Collecting grpc-google-iam-v1==0.14.2 (from -r /mount/src/chatbot-gaib/requirements.txt (line 74))

  Downloading grpc_google_iam_v1-0.14.2-py3-none-any.whl.metadata (9.1 kB)

Collecting grpcio==1.73.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 75))

  Downloading grpcio-1.73.0-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (3.8 kB)

Collecting grpcio-status==1.73.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 76))

  Downloading grpcio_status-1.73.0-py3-none-any.whl.metadata (1.1 kB)

Collecting h11==0.14.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 77))

  Downloading h11-0.14.0-py3-none-any.whl.metadata (8.2 kB)

Collecting httpcore==1.0.7 (from -r /mount/src/chatbot-gaib/requirements.txt (line 78))

  Downloading httpcore-1.0.7-py3-none-any.whl.metadata (21 kB)

Collecting httptools==0.6.4 (from -r /mount/src/chatbot-gaib/requirements.txt (line 79))

  Downloading httptools-0.6.4-cp313-cp313-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (3.6 kB)

Collecting httpx==0.28.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 80))

  Downloading httpx-0.28.1-py3-none-any.whl.metadata (7.1 kB)

Collecting httpx-sse==0.4.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 81))

  Downloading httpx_sse-0.4.0-py3-none-any.whl.metadata (9.0 kB)

Collecting huggingface-hub==0.33.4 (from -r /mount/src/chatbot-gaib/requirements.txt (line 82))

  Downloading huggingface_hub-0.33.4-py3-none-any.whl.metadata (14 kB)

Collecting humanfriendly==10.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 83))

  Downloading humanfriendly-10.0-py2.py3-none-any.whl.metadata (9.2 kB)

Collecting idna==3.10 (from -r /mount/src/chatbot-gaib/requirements.txt (line 84))

  Downloading idna-3.10-py3-none-any.whl.metadata (10 kB)

Collecting importlib_metadata==8.7.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 85))

  Downloading importlib_metadata-8.7.0-py3-none-any.whl.metadata (4.8 kB)

Collecting importlib_resources==6.5.2 (from -r /mount/src/chatbot-gaib/requirements.txt (line 86))

  Downloading importlib_resources-6.5.2-py3-none-any.whl.metadata (3.9 kB)

Collecting ipykernel==6.29.5 (from -r /mount/src/chatbot-gaib/requirements.txt (line 87))

  Downloading ipykernel-6.29.5-py3-none-any.whl.metadata (6.3 kB)

Collecting ipython==8.31.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 88))

  Downloading ipython-8.31.0-py3-none-any.whl.metadata (4.9 kB)

Collecting ipywidgets==8.1.5 (from -r /mount/src/chatbot-gaib/requirements.txt (line 89))

  Downloading ipywidgets-8.1.5-py3-none-any.whl.metadata (2.3 kB)

Collecting isoduration==20.11.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 90))

  Downloading isoduration-20.11.0-py3-none-any.whl.metadata (5.7 kB)

Collecting isort==6.0.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 91))

  Downloading isort-6.0.1-py3-none-any.whl.metadata (11 kB)

Collecting itsdangerous==2.2.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 92))

  Downloading itsdangerous-2.2.0-py3-none-any.whl.metadata (1.9 kB)

Collecting jedi==0.19.2 (from -r /mount/src/chatbot-gaib/requirements.txt (line 93))

  Downloading jedi-0.19.2-py2.py3-none-any.whl.metadata (22 kB)

Collecting Jinja2==3.1.5 (from -r /mount/src/chatbot-gaib/requirements.txt (line 94))

  Downloading jinja2-3.1.5-py3-none-any.whl.metadata (2.6 kB)

Collecting jiter==0.9.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 95))

  Downloading jiter-0.9.0-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (5.2 kB)

Collecting joblib==1.5.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 96))

  Downloading joblib-1.5.1-py3-none-any.whl.metadata (5.6 kB)

Collecting json5==0.10.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 97))

  Downloading json5-0.10.0-py3-none-any.whl.metadata (34 kB)

Collecting jsonpatch==1.33 (from -r /mount/src/chatbot-gaib/requirements.txt (line 98))

  Downloading jsonpatch-1.33-py2.py3-none-any.whl.metadata (3.0 kB)

Collecting jsonpointer==3.0.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 99))

  Downloading jsonpointer-3.0.0-py2.py3-none-any.whl.metadata (2.3 kB)

Collecting jsonschema==4.23.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 100))

  Downloading jsonschema-4.23.0-py3-none-any.whl.metadata (7.9 kB)

Collecting jsonschema-specifications==2024.10.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 101))

  Downloading jsonschema_specifications-2024.10.1-py3-none-any.whl.metadata (3.0 kB)

Collecting jupyter==1.1.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 102))

  Downloading jupyter-1.1.1-py2.py3-none-any.whl.metadata (2.0 kB)

Collecting jupyter-console==6.6.3 (from -r /mount/src/chatbot-gaib/requirements.txt (line 103))

  Downloading jupyter_console-6.6.3-py3-none-any.whl.metadata (5.8 kB)

Collecting jupyter-events==0.11.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 104))

  Downloading jupyter_events-0.11.0-py3-none-any.whl.metadata (5.8 kB)

Collecting jupyter-lsp==2.2.5 (from -r /mount/src/chatbot-gaib/requirements.txt (line 105))

  Downloading jupyter_lsp-2.2.5-py3-none-any.whl.metadata (1.8 kB)

Collecting jupyter_client==8.6.3 (from -r /mount/src/chatbot-gaib/requirements.txt (line 106))

  Downloading jupyter_client-8.6.3-py3-none-any.whl.metadata (8.3 kB)

Collecting jupyter_core==5.7.2 (from -r /mount/src/chatbot-gaib/requirements.txt (line 107))

  Downloading jupyter_core-5.7.2-py3-none-any.whl.metadata (3.4 kB)

Collecting jupyter_server==2.15.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 108))

  Downloading jupyter_server-2.15.0-py3-none-any.whl.metadata (8.4 kB)

Collecting jupyter_server_terminals==0.5.3 (from -r /mount/src/chatbot-gaib/requirements.txt (line 109))

  Downloading jupyter_server_terminals-0.5.3-py3-none-any.whl.metadata (5.6 kB)

Collecting jupyterlab==4.3.4 (from -r /mount/src/chatbot-gaib/requirements.txt (line 110))

  Downloading jupyterlab-4.3.4-py3-none-any.whl.metadata (16 kB)

Collecting jupyterlab_pygments==0.3.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 111))

  Downloading jupyterlab_pygments-0.3.0-py3-none-any.whl.metadata (4.4 kB)

Collecting jupyterlab_server==2.27.3 (from -r /mount/src/chatbot-gaib/requirements.txt (line 112))

  Downloading jupyterlab_server-2.27.3-py3-none-any.whl.metadata (5.9 kB)

Collecting jupyterlab_widgets==3.0.13 (from -r /mount/src/chatbot-gaib/requirements.txt (line 113))

  Downloading jupyterlab_widgets-3.0.13-py3-none-any.whl.metadata (4.1 kB)

Collecting kiwisolver==1.4.8 (from -r /mount/src/chatbot-gaib/requirements.txt (line 114))

  Downloading kiwisolver-1.4.8-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.2 kB)

Collecting kubernetes==33.1.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 115))

  Downloading kubernetes-33.1.0-py2.py3-none-any.whl.metadata (1.7 kB)

Collecting langchain==0.3.26 (from -r /mount/src/chatbot-gaib/requirements.txt (line 116))

  Downloading langchain-0.3.26-py3-none-any.whl.metadata (7.8 kB)

Collecting langchain-community==0.3.26 (from -r /mount/src/chatbot-gaib/requirements.txt (line 117))

  Downloading langchain_community-0.3.26-py3-none-any.whl.metadata (2.9 kB)

Collecting langchain-core==0.3.68 (from -r /mount/src/chatbot-gaib/requirements.txt (line 118))

  Downloading langchain_core-0.3.68-py3-none-any.whl.metadata (5.8 kB)

Collecting langchain-experimental==0.3.4 (from -r /mount/src/chatbot-gaib/requirements.txt (line 119))

  Downloading langchain_experimental-0.3.4-py3-none-any.whl.metadata (1.7 kB)

Collecting langchain-google-genai==2.1.5 (from -r /mount/src/chatbot-gaib/requirements.txt (line 120))

  Downloading langchain_google_genai-2.1.5-py3-none-any.whl.metadata (5.2 kB)

Collecting langchain-google-vertexai==2.0.27 (from -r /mount/src/chatbot-gaib/requirements.txt (line 121))

  Downloading langchain_google_vertexai-2.0.27-py3-none-any.whl.metadata (4.8 kB)

Collecting langchain-groq==0.3.2 (from -r /mount/src/chatbot-gaib/requirements.txt (line 122))

  Downloading langchain_groq-0.3.2-py3-none-any.whl.metadata (2.6 kB)

Collecting langchain-text-splitters==0.3.8 (from -r /mount/src/chatbot-gaib/requirements.txt (line 123))

  Downloading langchain_text_splitters-0.3.8-py3-none-any.whl.metadata (1.9 kB)

Collecting langsmith==0.3.45 (from -r /mount/src/chatbot-gaib/requirements.txt (line 124))

  Downloading langsmith-0.3.45-py3-none-any.whl.metadata (15 kB)

Collecting mando==0.7.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 125))

  Downloading mando-0.7.1-py2.py3-none-any.whl.metadata (7.4 kB)

Collecting markdown-it-py==3.0.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 126))

  Downloading markdown_it_py-3.0.0-py3-none-any.whl.metadata (6.9 kB)

Collecting MarkupSafe==3.0.2 (from -r /mount/src/chatbot-gaib/requirements.txt (line 127))

  Downloading MarkupSafe-3.0.2-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (4.0 kB)

Collecting marshmallow==3.26.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 128))

  Downloading marshmallow-3.26.1-py3-none-any.whl.metadata (7.3 kB)

Collecting matplotlib==3.10.3 (from -r /mount/src/chatbot-gaib/requirements.txt (line 129))

  Downloading matplotlib-3.10.3-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (11 kB)

Collecting matplotlib-inline==0.1.7 (from -r /mount/src/chatbot-gaib/requirements.txt (line 130))

  Downloading matplotlib_inline-0.1.7-py3-none-any.whl.metadata (3.9 kB)

Collecting mccabe==0.7.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 131))

  Downloading mccabe-0.7.0-py2.py3-none-any.whl.metadata (5.0 kB)

Collecting mdurl==0.1.2 (from -r /mount/src/chatbot-gaib/requirements.txt (line 132))

  Downloading mdurl-0.1.2-py3-none-any.whl.metadata (1.6 kB)

Collecting mistune==3.1.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 133))

  Downloading mistune-3.1.0-py3-none-any.whl.metadata (1.7 kB)

Collecting mmh3==5.1.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 134))

  Downloading mmh3-5.1.0-cp313-cp313-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (16 kB)

Collecting mpmath==1.3.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 135))

  Downloading mpmath-1.3.0-py3-none-any.whl.metadata (8.6 kB)

Collecting multidict==6.3.2 (from -r /mount/src/chatbot-gaib/requirements.txt (line 136))

  Downloading multidict-6.3.2-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (5.1 kB)

Collecting mypy_extensions==1.1.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 137))

  Downloading mypy_extensions-1.1.0-py3-none-any.whl.metadata (1.1 kB)

Collecting narwhals==1.43.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 138))

  Downloading narwhals-1.43.1-py3-none-any.whl.metadata (11 kB)

Collecting nbclient==0.10.2 (from -r /mount/src/chatbot-gaib/requirements.txt (line 139))

  Downloading nbclient-0.10.2-py3-none-any.whl.metadata (8.3 kB)

Collecting nbconvert==7.16.5 (from -r /mount/src/chatbot-gaib/requirements.txt (line 140))

  Downloading nbconvert-7.16.5-py3-none-any.whl.metadata (8.5 kB)

Collecting nbformat==5.10.4 (from -r /mount/src/chatbot-gaib/requirements.txt (line 141))

  Downloading nbformat-5.10.4-py3-none-any.whl.metadata (3.6 kB)

Collecting nest-asyncio==1.6.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 142))

  Downloading nest_asyncio-1.6.0-py3-none-any.whl.metadata (2.8 kB)

Collecting networkx==3.5 (from -r /mount/src/chatbot-gaib/requirements.txt (line 143))

  Downloading networkx-3.5-py3-none-any.whl.metadata (6.3 kB)

Collecting notebook==7.3.2 (from -r /mount/src/chatbot-gaib/requirements.txt (line 144))

  Downloading notebook-7.3.2-py3-none-any.whl.metadata (10 kB)

Collecting notebook_shim==0.2.4 (from -r /mount/src/chatbot-gaib/requirements.txt (line 145))

  Downloading notebook_shim-0.2.4-py3-none-any.whl.metadata (4.0 kB)

Collecting numexpr==2.11.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 146))

  Downloading numexpr-2.11.0-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (9.0 kB)

Collecting numpy==2.2.2 (from -r /mount/src/chatbot-gaib/requirements.txt (line 147))

  Downloading numpy-2.2.2-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (62 kB)

     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 62.0/62.0 kB 114.3 MB/s eta 0:00:00[2025-08-20 10:31:39.900142] 

Collecting oauthlib==3.3.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 148))

  Downloading oauthlib-3.3.1-py3-none-any.whl.metadata (7.9 kB)

Collecting onnxruntime==1.22.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 149))

  Downloading onnxruntime-1.22.1-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (4.9 kB)

Collecting openai==0.28.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 150))

  Downloading openai-0.28.0-py3-none-any.whl.metadata (13 kB)

Collecting openpyxl==3.1.5 (from -r /mount/src/chatbot-gaib/requirements.txt (line 151))

  Downloading openpyxl-3.1.5-py2.py3-none-any.whl.metadata (2.5 kB)

Collecting opentelemetry-api==1.35.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 152))

  Downloading opentelemetry_api-1.35.0-py3-none-any.whl.metadata (1.5 kB)

Collecting opentelemetry-exporter-otlp-proto-common==1.35.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 153))

  Downloading opentelemetry_exporter_otlp_proto_common-1.35.0-py3-none-any.whl.metadata (1.8 kB)

Collecting opentelemetry-exporter-otlp-proto-grpc==1.35.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 154))

  Downloading opentelemetry_exporter_otlp_proto_grpc-1.35.0-py3-none-any.whl.metadata (2.4 kB)

Collecting opentelemetry-proto==1.35.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 155))

  Downloading opentelemetry_proto-1.35.0-py3-none-any.whl.metadata (2.3 kB)

Collecting opentelemetry-sdk==1.35.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 156))

  Downloading opentelemetry_sdk-1.35.0-py3-none-any.whl.metadata (1.5 kB)

Collecting opentelemetry-semantic-conventions==0.56b0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 157))

  Downloading opentelemetry_semantic_conventions-0.56b0-py3-none-any.whl.metadata (2.4 kB)

Collecting oracledb==3.1.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 158))

  Downloading oracledb-3.1.0-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.manylinux_2_28_x86_64.whl.metadata (5.4 kB)

Collecting orjson==3.10.18 (from -r /mount/src/chatbot-gaib/requirements.txt (line 159))

  Downloading orjson-3.10.18-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (41 kB)

     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 41.9/41.9 kB 110.1 MB/s eta 0:00:00[2025-08-20 10:31:41.337066] 

Collecting overrides==7.7.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 160))

  Downloading overrides-7.7.0-py3-none-any.whl.metadata (5.8 kB)

Collecting packaging==24.2 (from -r /mount/src/chatbot-gaib/requirements.txt (line 161))

  Downloading packaging-24.2-py3-none-any.whl.metadata (3.2 kB)

Collecting pandas==2.3.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 162))

  Downloading pandas-2.3.1-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (91 kB)

     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 91.2/91.2 kB 180.2 MB/s eta 0:00:00[2025-08-20 10:31:41.682348] 

Collecting pandas-gbq==0.29.2 (from -r /mount/src/chatbot-gaib/requirements.txt (line 163))

  Downloading pandas_gbq-0.29.2-py3-none-any.whl.metadata (3.6 kB)

Collecting pandocfilters==1.5.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 164))

  Downloading pandocfilters-1.5.1-py2.py3-none-any.whl.metadata (9.0 kB)

Collecting parso==0.8.4 (from -r /mount/src/chatbot-gaib/requirements.txt (line 165))

  Downloading parso-0.8.4-py2.py3-none-any.whl.metadata (7.7 kB)

Collecting pathspec==0.12.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 166))

  Downloading pathspec-0.12.1-py3-none-any.whl.metadata (21 kB)

Collecting pillow==11.2.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 167))

  Downloading pillow-11.2.1-cp313-cp313-manylinux_2_28_x86_64.whl.metadata (8.9 kB)

Collecting pinecone-client==6.0.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 168))

  Downloading pinecone_client-6.0.0-py3-none-any.whl.metadata (3.4 kB)

Collecting pinecone-plugin-interface==0.0.7 (from -r /mount/src/chatbot-gaib/requirements.txt (line 169))

  Downloading pinecone_plugin_interface-0.0.7-py3-none-any.whl.metadata (1.2 kB)

Collecting platformdirs==4.3.6 (from -r /mount/src/chatbot-gaib/requirements.txt (line 170))

  Downloading platformdirs-4.3.6-py3-none-any.whl.metadata (11 kB)

Collecting plotly==6.2.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 171))

  Downloading plotly-6.2.0-py3-none-any.whl.metadata (8.5 kB)

Collecting posthog==5.4.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 172))

  Downloading posthog-5.4.0-py3-none-any.whl.metadata (5.7 kB)

Collecting prometheus_client==0.21.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 173))

  Downloading prometheus_client-0.21.1-py3-none-any.whl.metadata (1.8 kB)

Collecting prompt_toolkit==3.0.50 (from -r /mount/src/chatbot-gaib/requirements.txt (line 174))

  Downloading prompt_toolkit-3.0.50-py3-none-any.whl.metadata (6.6 kB)

Collecting propcache==0.3.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 175))

  Downloading propcache-0.3.1-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (10 kB)

Collecting proto-plus==1.26.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 176))

  Downloading proto_plus-1.26.1-py3-none-any.whl.metadata (2.2 kB)

Collecting protobuf==6.31.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 177))

  Downloading protobuf-6.31.1-cp39-abi3-manylinux2014_x86_64.whl.metadata (593 bytes)

Collecting psutil==6.1.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 178))

  Downloading psutil-6.1.1-cp36-abi3-manylinux_2_12_x86_64.manylinux2010_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (22 kB)

Collecting pure_eval==0.2.3 (from -r /mount/src/chatbot-gaib/requirements.txt (line 179))

  Downloading pure_eval-0.2.3-py3-none-any.whl.metadata (6.3 kB)

Collecting pyarrow==19.0.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 180))

  Downloading pyarrow-19.0.1-cp313-cp313-manylinux_2_28_x86_64.whl.metadata (3.3 kB)

Collecting pyasn1==0.6.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 181))

  Downloading pyasn1-0.6.1-py3-none-any.whl.metadata (8.4 kB)

Collecting pyasn1_modules==0.4.2 (from -r /mount/src/chatbot-gaib/requirements.txt (line 182))

  Downloading pyasn1_modules-0.4.2-py3-none-any.whl.metadata (3.5 kB)

Collecting pybase64==1.4.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 183))

  Downloading pybase64-1.4.1-cp313-cp313-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (8.4 kB)

Collecting pycparser==2.22 (from -r /mount/src/chatbot-gaib/requirements.txt (line 184))

  Downloading pycparser-2.22-py3-none-any.whl.metadata (943 bytes)

Collecting pydantic==2.11.2 (from -r /mount/src/chatbot-gaib/requirements.txt (line 185))

  Downloading pydantic-2.11.2-py3-none-any.whl.metadata (64 kB)

     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 64.7/64.7 kB 165.9 MB/s eta 0:00:00[2025-08-20 10:31:44.505513] 

Collecting pydantic-settings==2.10.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 186))

  Downloading pydantic_settings-2.10.0-py3-none-any.whl.metadata (3.4 kB)

Collecting pydantic_core==2.33.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 187))

  Downloading pydantic_core-2.33.1-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.8 kB)

Collecting pydata-google-auth==1.9.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 188))

  Downloading pydata_google_auth-1.9.1-py2.py3-none-any.whl.metadata (2.8 kB)

Collecting pydeck==0.9.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 189))

  Downloading pydeck-0.9.1-py2.py3-none-any.whl.metadata (4.1 kB)

Collecting Pygments==2.19.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 190))

  Downloading pygments-2.19.1-py3-none-any.whl.metadata (2.5 kB)

Collecting pylint==3.3.7 (from -r /mount/src/chatbot-gaib/requirements.txt (line 191))

  Downloading pylint-3.3.7-py3-none-any.whl.metadata (12 kB)

Collecting pyparsing==3.2.3 (from -r /mount/src/chatbot-gaib/requirements.txt (line 192))

  Downloading pyparsing-3.2.3-py3-none-any.whl.metadata (5.0 kB)

Collecting pypdf==5.8.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 193))

  Downloading pypdf-5.8.0-py3-none-any.whl.metadata (7.1 kB)

Collecting PyPika==0.48.9 (from -r /mount/src/chatbot-gaib/requirements.txt (line 194))

  Downloading PyPika-0.48.9.tar.gz (67 kB)

     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 67.3/67.3 kB 116.7 MB/s eta 0:00:00[2025-08-20 10:31:46.371962] 

  Installing build dependencies: started

  Installing build dependencies: finished with status 'done'

  Getting requirements to build wheel: started

  Getting requirements to build wheel: finished with status 'done'

  Preparing metadata (pyproject.toml): started

  Preparing metadata (pyproject.toml): finished with status 'done'

Collecting pyproject_hooks==1.2.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 195))

  Downloading pyproject_hooks-1.2.0-py3-none-any.whl.metadata (1.3 kB)

Collecting pyreadline3==3.5.4 (from -r /mount/src/chatbot-gaib/requirements.txt (line 196))

  Downloading pyreadline3-3.5.4-py3-none-any.whl.metadata (4.7 kB)

Collecting python-dateutil==2.9.0.post0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 197))

  Downloading python_dateutil-2.9.0.post0-py2.py3-none-any.whl.metadata (8.4 kB)

Collecting python-dotenv==1.1.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 198))

  Downloading python_dotenv-1.1.0-py3-none-any.whl.metadata (24 kB)

Collecting python-json-logger==3.2.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 199))

  Downloading python_json_logger-3.2.1-py3-none-any.whl.metadata (4.1 kB)

Collecting pytz==2024.2 (from -r /mount/src/chatbot-gaib/requirements.txt (line 200))

  Downloading pytz-2024.2-py2.py3-none-any.whl.metadata (22 kB)

Collecting pywinpty==2.0.14 (from -r /mount/src/chatbot-gaib/requirements.txt (line 201))

  Downloading pywinpty-2.0.14.tar.gz (27 kB)

  Installing build dependencies: started

  Installing build dependencies: finished with status 'done'

  Getting requirements to build wheel: started

  Getting requirements to build wheel: finished with status 'done'

  Installing backend dependencies: started

  Installing backend dependencies: finished with status 'done'

  Preparing metadata (pyproject.toml): started

  Preparing metadata (pyproject.toml): finished with status 'done'

Collecting PyYAML==6.0.2 (from -r /mount/src/chatbot-gaib/requirements.txt (line 202))

  Downloading PyYAML-6.0.2-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (2.1 kB)

Collecting pyzmq==26.2.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 203))

  Downloading pyzmq-26.2.0-cp313-cp313-manylinux_2_28_x86_64.whl.metadata (6.2 kB)

Collecting radon==6.0.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 204))

  Downloading radon-6.0.1-py2.py3-none-any.whl.metadata (8.2 kB)

Collecting referencing==0.36.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 205))

  Downloading referencing-0.36.1-py3-none-any.whl.metadata (2.8 kB)

Collecting regex==2024.11.6 (from -r /mount/src/chatbot-gaib/requirements.txt (line 206))

  Downloading regex-2024.11.6-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (40 kB)

     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 40.5/40.5 kB 134.0 MB/s eta 0:00:00[2025-08-20 10:31:57.511815] 

Collecting requests==2.32.4 (from -r /mount/src/chatbot-gaib/requirements.txt (line 207))

  Downloading requests-2.32.4-py3-none-any.whl.metadata (4.9 kB)

Collecting requests-oauthlib==2.0.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 208))

  Downloading requests_oauthlib-2.0.0-py2.py3-none-any.whl.metadata (11 kB)

Collecting requests-toolbelt==1.0.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 209))

  Downloading requests_toolbelt-1.0.0-py2.py3-none-any.whl.metadata (14 kB)

Collecting rfc3339-validator==0.1.4 (from -r /mount/src/chatbot-gaib/requirements.txt (line 210))

  Downloading rfc3339_validator-0.1.4-py2.py3-none-any.whl.metadata (1.5 kB)

Collecting rfc3986-validator==0.1.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 211))

  Downloading rfc3986_validator-0.1.1-py2.py3-none-any.whl.metadata (1.7 kB)

Collecting rich==14.0.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 212))

  Downloading rich-14.0.0-py3-none-any.whl.metadata (18 kB)

Collecting rpds-py==0.22.3 (from -r /mount/src/chatbot-gaib/requirements.txt (line 213))

  Downloading rpds_py-0.22.3-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (4.2 kB)

Collecting rsa==4.9.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 214))

  Downloading rsa-4.9.1-py3-none-any.whl.metadata (5.6 kB)

Collecting safetensors==0.5.3 (from -r /mount/src/chatbot-gaib/requirements.txt (line 215))

  Downloading safetensors-0.5.3-cp38-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (3.8 kB)

Collecting scikit-learn==1.7.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 216))

  Downloading scikit_learn-1.7.1-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (11 kB)

Collecting scipy==1.16.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 217))

  Downloading scipy-1.16.0-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (61 kB)

     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 61.9/61.9 kB 161.8 MB/s eta 0:00:00[2025-08-20 10:31:59.256276] 

Collecting seaborn==0.13.2 (from -r /mount/src/chatbot-gaib/requirements.txt (line 218))

  Downloading seaborn-0.13.2-py3-none-any.whl.metadata (5.4 kB)

Collecting Send2Trash==1.8.3 (from -r /mount/src/chatbot-gaib/requirements.txt (line 219))

  Downloading Send2Trash-1.8.3-py3-none-any.whl.metadata (4.0 kB)

Collecting sentence-transformers==5.0.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 220))

  Downloading sentence_transformers-5.0.0-py3-none-any.whl.metadata (16 kB)

Collecting setuptools==75.8.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 221))

  Downloading setuptools-75.8.0-py3-none-any.whl.metadata (6.7 kB)

Collecting shapely==2.1.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 222))

  Downloading shapely-2.1.1-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.8 kB)

Collecting shellingham==1.5.4 (from -r /mount/src/chatbot-gaib/requirements.txt (line 223))

  Downloading shellingham-1.5.4-py2.py3-none-any.whl.metadata (3.5 kB)

Collecting six==1.17.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 224))

  Downloading six-1.17.0-py2.py3-none-any.whl.metadata (1.7 kB)

Collecting smmap==5.0.2 (from -r /mount/src/chatbot-gaib/requirements.txt (line 225))

  Downloading smmap-5.0.2-py3-none-any.whl.metadata (4.3 kB)

Collecting sniffio==1.3.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 226))

  Downloading sniffio-1.3.1-py3-none-any.whl.metadata (3.9 kB)

Collecting soupsieve==2.6 (from -r /mount/src/chatbot-gaib/requirements.txt (line 227))

  Downloading soupsieve-2.6-py3-none-any.whl.metadata (4.6 kB)

Collecting SQLAlchemy==2.0.41 (from -r /mount/src/chatbot-gaib/requirements.txt (line 228))

  Downloading sqlalchemy-2.0.41-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (9.6 kB)

Collecting sqlalchemy_bigquery==0.0.7 (from -r /mount/src/chatbot-gaib/requirements.txt (line 229))

  Downloading sqlalchemy_bigquery-0.0.7.tar.gz (5.1 kB)

  Installing build dependencies: started

  Installing build dependencies: finished with status 'done'

  Getting requirements to build wheel: started

  Getting requirements to build wheel: finished with status 'done'

  Preparing metadata (pyproject.toml): started

  Preparing metadata (pyproject.toml): finished with status 'done'

Collecting sqlparse==0.5.3 (from -r /mount/src/chatbot-gaib/requirements.txt (line 230))

  Downloading sqlparse-0.5.3-py3-none-any.whl.metadata (3.9 kB)

Collecting stack-data==0.6.3 (from -r /mount/src/chatbot-gaib/requirements.txt (line 231))

  Downloading stack_data-0.6.3-py3-none-any.whl.metadata (18 kB)

Collecting streamlit==1.46.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 232))

  Downloading streamlit-1.46.0-py3-none-any.whl.metadata (9.0 kB)

Collecting sympy==1.14.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 233))

  Downloading sympy-1.14.0-py3-none-any.whl.metadata (12 kB)

Collecting tabulate==0.9.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 234))

  Downloading tabulate-0.9.0-py3-none-any.whl.metadata (34 kB)

Collecting tenacity==8.5.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 235))

  Downloading tenacity-8.5.0-py3-none-any.whl.metadata (1.2 kB)

Collecting terminado==0.18.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 236))

  Downloading terminado-0.18.1-py3-none-any.whl.metadata (5.8 kB)

Collecting threadpoolctl==3.6.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 237))

  Downloading threadpoolctl-3.6.0-py3-none-any.whl.metadata (13 kB)

Collecting tinycss2==1.4.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 238))

  Downloading tinycss2-1.4.0-py3-none-any.whl.metadata (3.0 kB)

Collecting tokenizers==0.21.2 (from -r /mount/src/chatbot-gaib/requirements.txt (line 239))

  Downloading tokenizers-0.21.2-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.8 kB)

Collecting toml==0.10.2 (from -r /mount/src/chatbot-gaib/requirements.txt (line 240))

  Downloading toml-0.10.2-py2.py3-none-any.whl.metadata (7.1 kB)

Collecting tomlkit==0.13.3 (from -r /mount/src/chatbot-gaib/requirements.txt (line 241))

  Downloading tomlkit-0.13.3-py3-none-any.whl.metadata (2.8 kB)

Collecting torch==2.7.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 242))

  Downloading torch-2.7.1-cp313-cp313-manylinux_2_28_x86_64.whl.metadata (29 kB)

Collecting tornado==6.4.2 (from -r /mount/src/chatbot-gaib/requirements.txt (line 243))

  Downloading tornado-6.4.2-cp38-abi3-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (2.5 kB)

Collecting tqdm==4.67.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 244))

  Downloading tqdm-4.67.1-py3-none-any.whl.metadata (57 kB)

     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 57.7/57.7 kB 135.7 MB/s eta 0:00:00[2025-08-20 10:32:05.098441] 

Collecting traitlets==5.14.3 (from -r /mount/src/chatbot-gaib/requirements.txt (line 245))

  Downloading traitlets-5.14.3-py3-none-any.whl.metadata (10 kB)

Collecting transformers==4.53.2 (from -r /mount/src/chatbot-gaib/requirements.txt (line 246))

  Downloading transformers-4.53.2-py3-none-any.whl.metadata (40 kB)

     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 40.9/40.9 kB 164.0 MB/s eta 0:00:00[2025-08-20 10:32:05.247898] 

Collecting typer==0.16.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 247))

  Downloading typer-0.16.0-py3-none-any.whl.metadata (15 kB)

Collecting types-python-dateutil==2.9.0.20241206 (from -r /mount/src/chatbot-gaib/requirements.txt (line 248))

  Downloading types_python_dateutil-2.9.0.20241206-py3-none-any.whl.metadata (2.1 kB)

Collecting typing-inspect==0.9.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 249))

  Downloading typing_inspect-0.9.0-py3-none-any.whl.metadata (1.5 kB)

Collecting typing-inspection==0.4.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 250))

  Downloading typing_inspection-0.4.0-py3-none-any.whl.metadata (2.6 kB)

Collecting typing_extensions==4.12.2 (from -r /mount/src/chatbot-gaib/requirements.txt (line 251))

  Downloading typing_extensions-4.12.2-py3-none-any.whl.metadata (3.0 kB)

Collecting tzdata==2025.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 252))

  Downloading tzdata-2025.1-py2.py3-none-any.whl.metadata (1.4 kB)

Collecting uri-template==1.3.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 253))

  Downloading uri_template-1.3.0-py3-none-any.whl.metadata (8.8 kB)

Collecting urllib3==2.5.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 254))

  Downloading urllib3-2.5.0-py3-none-any.whl.metadata (6.5 kB)

Collecting uvicorn==0.35.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 255))

  Downloading uvicorn-0.35.0-py3-none-any.whl.metadata (6.5 kB)

Collecting validators==0.35.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 256))

  Downloading validators-0.35.0-py3-none-any.whl.metadata (3.9 kB)

Collecting watchdog==6.0.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 257))

  Downloading watchdog-6.0.0-py3-none-manylinux2014_x86_64.whl.metadata (44 kB)

     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 44.3/44.3 kB 111.5 MB/s eta 0:00:00[2025-08-20 10:32:05.852171] 

Collecting watchfiles==1.1.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 258))

  Downloading watchfiles-1.1.0-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (4.9 kB)

Collecting wcwidth==0.2.13 (from -r /mount/src/chatbot-gaib/requirements.txt (line 259))

  Downloading wcwidth-0.2.13-py2.py3-none-any.whl.metadata (14 kB)

Collecting webcolors==24.11.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 260))

  Downloading webcolors-24.11.1-py3-none-any.whl.metadata (2.2 kB)

Collecting webencodings==0.5.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 261))

  Downloading webencodings-0.5.1-py2.py3-none-any.whl.metadata (2.1 kB)

Collecting websocket-client==1.8.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 262))

  Downloading websocket_client-1.8.0-py3-none-any.whl.metadata (8.0 kB)

Collecting websockets==15.0.1 (from -r /mount/src/chatbot-gaib/requirements.txt (line 263))

  Downloading websockets-15.0.1-cp313-cp313-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.8 kB)

Collecting Werkzeug==3.1.3 (from -r /mount/src/chatbot-gaib/requirements.txt (line 264))

  Downloading werkzeug-3.1.3-py3-none-any.whl.metadata (3.7 kB)

Collecting widgetsnbextension==4.0.13 (from -r /mount/src/chatbot-gaib/requirements.txt (line 265))

  Downloading widgetsnbextension-4.0.13-py3-none-any.whl.metadata (1.6 kB)

Collecting yarl==1.19.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 266))

  Downloading yarl-1.19.0-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (71 kB)

     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 71.8/71.8 kB 139.2 MB/s eta 0:00:00[2025-08-20 10:32:06.845303] 

Collecting zipp==3.23.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 267))

  Downloading zipp-3.23.0-py3-none-any.whl.metadata (3.6 kB)

Collecting zstandard==0.23.0 (from -r /mount/src/chatbot-gaib/requirements.txt (line 268))

  Downloading zstandard-0.23.0-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (3.0 kB)

Collecting hf-xet<2.0.0,>=1.1.2 (from huggingface-hub==0.33.4->-r /mount/src/chatbot-gaib/requirements.txt (line 82))

  Downloading hf_xet-1.1.8-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (703 bytes)

Collecting pexpect>4.3 (from ipython==8.31.0->-r /mount/src/chatbot-gaib/requirements.txt (line 88))

  Downloading pexpect-4.9.0-py2.py3-none-any.whl.metadata (2.5 kB)

Collecting ptyprocess (from terminado==0.18.1->-r /mount/src/chatbot-gaib/requirements.txt (line 236))

  Downloading ptyprocess-0.7.0-py2.py3-none-any.whl.metadata (1.3 kB)

Collecting nvidia-cuda-nvrtc-cu12==12.6.77 (from torch==2.7.1->-r /mount/src/chatbot-gaib/requirements.txt (line 242))

  Downloading nvidia_cuda_nvrtc_cu12-12.6.77-py3-none-manylinux2014_x86_64.whl.metadata (1.5 kB)

Collecting nvidia-cuda-runtime-cu12==12.6.77 (from torch==2.7.1->-r /mount/src/chatbot-gaib/requirements.txt (line 242))

  Downloading nvidia_cuda_runtime_cu12-12.6.77-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.5 kB)

Collecting nvidia-cuda-cupti-cu12==12.6.80 (from torch==2.7.1->-r /mount/src/chatbot-gaib/requirements.txt (line 242))

  Downloading nvidia_cuda_cupti_cu12-12.6.80-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.6 kB)

Collecting nvidia-cudnn-cu12==9.5.1.17 (from torch==2.7.1->-r /mount/src/chatbot-gaib/requirements.txt (line 242))

  Downloading nvidia_cudnn_cu12-9.5.1.17-py3-none-manylinux_2_28_x86_64.whl.metadata (1.6 kB)

Collecting nvidia-cublas-cu12==12.6.4.1 (from torch==2.7.1->-r /mount/src/chatbot-gaib/requirements.txt (line 242))

  Downloading nvidia_cublas_cu12-12.6.4.1-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.5 kB)

Collecting nvidia-cufft-cu12==11.3.0.4 (from torch==2.7.1->-r /mount/src/chatbot-gaib/requirements.txt (line 242))

  Downloading nvidia_cufft_cu12-11.3.0.4-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.5 kB)

Collecting nvidia-curand-cu12==10.3.7.77 (from torch==2.7.1->-r /mount/src/chatbot-gaib/requirements.txt (line 242))

  Downloading nvidia_curand_cu12-10.3.7.77-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.5 kB)

Collecting nvidia-cusolver-cu12==11.7.1.2 (from torch==2.7.1->-r /mount/src/chatbot-gaib/requirements.txt (line 242))

  Downloading nvidia_cusolver_cu12-11.7.1.2-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.6 kB)

Collecting nvidia-cusparse-cu12==12.5.4.2 (from torch==2.7.1->-r /mount/src/chatbot-gaib/requirements.txt (line 242))

  Downloading nvidia_cusparse_cu12-12.5.4.2-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.6 kB)

Collecting nvidia-cusparselt-cu12==0.6.3 (from torch==2.7.1->-r /mount/src/chatbot-gaib/requirements.txt (line 242))

  Downloading nvidia_cusparselt_cu12-0.6.3-py3-none-manylinux2014_x86_64.whl.metadata (6.8 kB)

Collecting nvidia-nccl-cu12==2.26.2 (from torch==2.7.1->-r /mount/src/chatbot-gaib/requirements.txt (line 242))

  Downloading nvidia_nccl_cu12-2.26.2-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (2.0 kB)

Collecting nvidia-nvtx-cu12==12.6.77 (from torch==2.7.1->-r /mount/src/chatbot-gaib/requirements.txt (line 242))

  Downloading nvidia_nvtx_cu12-12.6.77-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.6 kB)

Collecting nvidia-nvjitlink-cu12==12.6.85 (from torch==2.7.1->-r /mount/src/chatbot-gaib/requirements.txt (line 242))

  Downloading nvidia_nvjitlink_cu12-12.6.85-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl.metadata (1.5 kB)

Collecting nvidia-cufile-cu12==1.11.1.6 (from torch==2.7.1->-r /mount/src/chatbot-gaib/requirements.txt (line 242))

  Downloading nvidia_cufile_cu12-1.11.1.6-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.5 kB)

Collecting triton==3.3.1 (from torch==2.7.1->-r /mount/src/chatbot-gaib/requirements.txt (line 242))

  Downloading triton-3.3.1-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (1.5 kB)

Collecting uvloop>=0.15.1 (from uvicorn[standard]>=0.18.3->chromadb==1.0.15->-r /mount/src/chatbot-gaib/requirements.txt (line 27))

  Downloading uvloop-0.21.0-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (4.9 kB)

WARNING: The candidate selected for download or install is a yanked version: 'multidict' candidate (version 6.3.2 at https://files.pythonhosted.org/packages/28/f8/18c81f5c5b7453dd8d15dc61ceca23d03c55e69f1937842039be2d8c4428/multidict-6.3.2-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (from https://pypi.org/simple/multidict/) (requires-python:>=3.9))

Reason for being yanked: Memory leak: https://github.com/aio-libs/multidict/issues/1131#issuecomment-2787514071

Downloading aiohappyeyeballs-2.6.1-py3-none-any.whl (15 kB)

Downloading aiohttp-3.11.16-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.7 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.7/1.7 MB 87.2 MB/s eta 0:00:00[2025-08-20 10:32:30.589055] 

Downloading aiosignal-1.3.2-py2.py3-none-any.whl (7.6 kB)

Downloading altair-5.5.0-py3-none-any.whl (731 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 731.2/731.2 kB 166.3 MB/s eta 0:00:00[2025-08-20 10:32:30.612812] 

Downloading annotated_types-0.7.0-py3-none-any.whl (13 kB)

Downloading anyio-4.8.0-py3-none-any.whl (96 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 96.0/96.0 kB 176.5 MB/s eta 0:00:00[2025-08-20 10:32:30.632664] 

Downloading argon2_cffi-23.1.0-py3-none-any.whl (15 kB)

Downloading argon2_cffi_bindings-21.2.0-cp36-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (86 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 86.2/86.2 kB 169.4 MB/s eta 0:00:00[2025-08-20 10:32:30.652549] 

Downloading arrow-1.3.0-py3-none-any.whl (66 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 66.4/66.4 kB 178.4 MB/s eta 0:00:00[2025-08-20 10:32:30.663896] 

Downloading astroid-3.3.10-py3-none-any.whl (275 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 275.4/275.4 kB 213.6 MB/s eta 0:00:00[2025-08-20 10:32:30.676928] 

Downloading asttokens-3.0.0-py3-none-any.whl (26 kB)

Downloading async_lru-2.0.4-py3-none-any.whl (6.1 kB)

Downloading attrs-24.3.0-py3-none-any.whl (63 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 63.4/63.4 kB 150.7 MB/s eta 0:00:00[2025-08-20 10:32:30.706212] 

Downloading babel-2.16.0-py3-none-any.whl (9.6 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 9.6/9.6 MB 128.9 MB/s eta 0:00:00[2025-08-20 10:32:30.796708] 

Downloading backoff-2.2.1-py3-none-any.whl (15 kB)

Downloading bcrypt-4.3.0-cp39-abi3-manylinux_2_28_x86_64.whl (284 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 284.5/284.5 kB 131.0 MB/s eta 0:00:00[2025-08-20 10:32:30.821532] 

Downloading beautifulsoup4-4.12.3-py3-none-any.whl (147 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 147.9/147.9 kB 183.8 MB/s eta 0:00:00[2025-08-20 10:32:30.835188] 

Downloading black-25.1.0-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.manylinux_2_28_x86_64.whl (1.8 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.8/1.8 MB 124.5 MB/s eta 0:00:00[2025-08-20 10:32:30.873714] 

Downloading bleach-6.2.0-py3-none-any.whl (163 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 163.4/163.4 kB 151.7 MB/s eta 0:00:00[2025-08-20 10:32:30.886804] 

Downloading blinker-1.9.0-py3-none-any.whl (8.5 kB)

Downloading bottleneck-1.5.0-cp313-cp313-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (362 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 362.8/362.8 kB 149.0 MB/s eta 0:00:00[2025-08-20 10:32:30.913403] 

Downloading build-1.2.2.post1-py3-none-any.whl (22 kB)

Downloading cachetools-5.5.2-py3-none-any.whl (10 kB)

Downloading certifi-2025.6.15-py3-none-any.whl (157 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 157.7/157.7 kB 157.8 MB/s eta 0:00:00[2025-08-20 10:32:30.943908] 

Downloading cffi-1.17.1-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (479 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 479.4/479.4 kB 189.6 MB/s eta 0:00:00[2025-08-20 10:32:30.958074] 

Downloading charset_normalizer-3.4.2-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (148 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 148.2/148.2 kB 143.4 MB/s eta 0:00:00[2025-08-20 10:32:30.971908] 

Downloading chromadb-1.0.15-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (19.5 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 19.5/19.5 MB 194.2 MB/s eta 0:00:00[2025-08-20 10:32:31.088989] 

Downloading click-8.1.8-py3-none-any.whl (98 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 98.2/98.2 kB 174.4 MB/s eta 0:00:00[2025-08-20 10:32:31.100093] 

Downloading colorama-0.4.6-py2.py3-none-any.whl (25 kB)

Downloading coloredlogs-15.0.1-py2.py3-none-any.whl (46 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 46.0/46.0 kB 129.6 MB/s eta 0:00:00[2025-08-20 10:32:31.119763] 

Downloading comm-0.2.2-py3-none-any.whl (7.2 kB)

Downloading contourpy-1.3.2-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (322 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 322.7/322.7 kB 206.0 MB/s eta 0:00:00[2025-08-20 10:32:31.143049] 

Downloading cryptography-44.0.2-cp39-abi3-manylinux_2_28_x86_64.whl (4.2 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.2/4.2 MB 184.5 MB/s eta 0:00:00[2025-08-20 10:32:31.179630] 

Downloading cycler-0.12.1-py3-none-any.whl (8.3 kB)

Downloading dataclasses_json-0.6.7-py3-none-any.whl (28 kB)

Downloading db_dtypes-1.4.3-py3-none-any.whl (18 kB)

Downloading debugpy-1.8.12-cp313-cp313-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (4.2 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.2/4.2 MB 210.5 MB/s eta 0:00:00[2025-08-20 10:32:31.240718] 

Downloading decorator-5.1.1-py3-none-any.whl (9.1 kB)

Downloading defusedxml-0.7.1-py2.py3-none-any.whl (25 kB)

Downloading dill-0.4.0-py3-none-any.whl (119 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 119.7/119.7 kB 132.5 MB/s eta 0:00:00[2025-08-20 10:32:31.270081] 

Downloading distro-1.9.0-py3-none-any.whl (20 kB)

Downloading docstring_parser-0.16-py3-none-any.whl (36 kB)

Downloading durationpy-0.10-py3-none-any.whl (3.9 kB)

Downloading et_xmlfile-2.0.0-py3-none-any.whl (18 kB)

Downloading executing-2.2.0-py2.py3-none-any.whl (26 kB)

Downloading Faker-35.0.0-py3-none-any.whl (1.9 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.9/1.9 MB 143.8 MB/s eta 0:00:00[2025-08-20 10:32:31.340618] 

Downloading fastjsonschema-2.21.1-py3-none-any.whl (23 kB)

Downloading filelock-3.18.0-py3-none-any.whl (16 kB)

Downloading filetype-1.2.0-py2.py3-none-any.whl (19 kB)

Downloading flask-3.1.0-py3-none-any.whl (102 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 103.0/103.0 kB 121.9 MB/s eta 0:00:00[2025-08-20 10:32:31.381654] 

Downloading flatbuffers-25.2.10-py2.py3-none-any.whl (30 kB)

Downloading fonttools-4.59.0-cp313-cp313-manylinux1_x86_64.manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_5_x86_64.whl (4.9 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.9/4.9 MB 128.8 MB/s eta 0:00:00[2025-08-20 10:32:31.444925] 

Downloading fqdn-1.5.1-py3-none-any.whl (9.1 kB)

Downloading frozenlist-1.5.0-cp313-cp313-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (267 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 267.0/267.0 kB 134.1 MB/s eta 0:00:00[2025-08-20 10:32:31.469110] 

Downloading fsspec-2025.7.0-py3-none-any.whl (199 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 199.6/199.6 kB 134.8 MB/s eta 0:00:00[2025-08-20 10:32:31.483284] 

Downloading gitdb-4.0.12-py3-none-any.whl (62 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 62.8/62.8 kB 117.7 MB/s eta 0:00:00[2025-08-20 10:32:31.496181] 

Downloading GitPython-3.1.44-py3-none-any.whl (207 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 207.6/207.6 kB 138.2 MB/s eta 0:00:00[2025-08-20 10:32:31.510021] 

Downloading google_ai_generativelanguage-0.6.18-py3-none-any.whl (1.4 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.4/1.4 MB 152.3 MB/s eta 0:00:00[2025-08-20 10:32:31.533814] 

Downloading google_api_core-2.25.1-py3-none-any.whl (160 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 160.8/160.8 kB 130.9 MB/s eta 0:00:00[2025-08-20 10:32:31.547678] 

Downloading google_auth-2.40.3-py2.py3-none-any.whl (216 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 216.1/216.1 kB 122.9 MB/s eta 0:00:00[2025-08-20 10:32:31.561915] 

Downloading google_auth_oauthlib-1.2.2-py3-none-any.whl (19 kB)

Downloading google_cloud_aiplatform-1.105.0-py2.py3-none-any.whl (7.9 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7.9/7.9 MB 142.1 MB/s eta 0:00:00[2025-08-20 10:32:31.641622] 

Downloading google_cloud_bigquery-3.34.0-py3-none-any.whl (253 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 253.6/253.6 kB 162.0 MB/s eta 0:00:00[2025-08-20 10:32:31.656730] 

Downloading google_cloud_bigquery_storage-2.32.0-py3-none-any.whl (296 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 296.4/296.4 kB 179.0 MB/s eta 0:00:00[2025-08-20 10:32:31.671610] 

Downloading google_cloud_core-2.4.3-py2.py3-none-any.whl (29 kB)

Downloading google_cloud_resource_manager-1.14.2-py3-none-any.whl (394 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 394.3/394.3 kB 205.4 MB/s eta 0:00:00[2025-08-20 10:32:31.697348] 

Downloading google_cloud_storage-2.19.0-py2.py3-none-any.whl (131 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 131.8/131.8 kB 142.6 MB/s eta 0:00:00[2025-08-20 10:32:31.710426] 

Downloading google_crc32c-1.7.1-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (32 kB)

Downloading google_genai-1.21.1-py3-none-any.whl (206 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 206.4/206.4 kB 215.1 MB/s eta 0:00:00[2025-08-20 10:32:31.732880] 

Downloading google_resumable_media-2.7.2-py2.py3-none-any.whl (81 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 81.3/81.3 kB 140.0 MB/s eta 0:00:00[2025-08-20 10:32:31.744533] 

Downloading googleapis_common_protos-1.70.0-py3-none-any.whl (294 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 294.5/294.5 kB 206.1 MB/s eta 0:00:00[2025-08-20 10:32:31.756610] 

Downloading greenlet-3.2.3-cp313-cp313-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl (608 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 608.4/608.4 kB 132.6 MB/s eta 0:00:00[2025-08-20 10:32:31.776613] 

Downloading groq-0.28.0-py3-none-any.whl (130 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 130.2/130.2 kB 181.6 MB/s eta 0:00:00[2025-08-20 10:32:31.790417] 

Downloading grpc_google_iam_v1-0.14.2-py3-none-any.whl (19 kB)

Downloading grpcio-1.73.0-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (6.0 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 6.0/6.0 MB 155.1 MB/s eta 0:00:00[2025-08-20 10:32:31.856271] 

Downloading grpcio_status-1.73.0-py3-none-any.whl (14 kB)

Downloading h11-0.14.0-py3-none-any.whl (58 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 58.3/58.3 kB 111.4 MB/s eta 0:00:00[2025-08-20 10:32:31.879169] 

Downloading httpcore-1.0.7-py3-none-any.whl (78 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 78.6/78.6 kB 119.6 MB/s eta 0:00:00[2025-08-20 10:32:31.891979] 

Downloading httptools-0.6.4-cp313-cp313-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (473 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 473.8/473.8 kB 129.6 MB/s eta 0:00:00[2025-08-20 10:32:31.914723] 

Downloading httpx-0.28.1-py3-none-any.whl (73 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 73.5/73.5 kB 123.6 MB/s eta 0:00:00[2025-08-20 10:32:31.927765] 

Downloading httpx_sse-0.4.0-py3-none-any.whl (7.8 kB)

Downloading huggingface_hub-0.33.4-py3-none-any.whl (515 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 515.3/515.3 kB 132.3 MB/s eta 0:00:00[2025-08-20 10:32:31.954338] 

Downloading humanfriendly-10.0-py2.py3-none-any.whl (86 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 86.8/86.8 kB 121.7 MB/s eta 0:00:00[2025-08-20 10:32:31.966721] 

Downloading idna-3.10-py3-none-any.whl (70 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 70.4/70.4 kB 120.4 MB/s eta 0:00:00[2025-08-20 10:32:31.978529] 

Downloading importlib_metadata-8.7.0-py3-none-any.whl (27 kB)

Downloading importlib_resources-6.5.2-py3-none-any.whl (37 kB)

Downloading ipykernel-6.29.5-py3-none-any.whl (117 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 117.2/117.2 kB 128.1 MB/s eta 0:00:00[2025-08-20 10:32:32.009288] 

Downloading ipython-8.31.0-py3-none-any.whl (821 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 821.6/821.6 kB 135.4 MB/s eta 0:00:00[2025-08-20 10:32:32.031062] 

Downloading ipywidgets-8.1.5-py3-none-any.whl (139 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 139.8/139.8 kB 130.6 MB/s eta 0:00:00[2025-08-20 10:32:32.045185] 

Downloading isoduration-20.11.0-py3-none-any.whl (11 kB)

Downloading isort-6.0.1-py3-none-any.whl (94 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 94.2/94.2 kB 128.3 MB/s eta 0:00:00[2025-08-20 10:32:32.066470] 

Downloading itsdangerous-2.2.0-py3-none-any.whl (16 kB)

Downloading jedi-0.19.2-py2.py3-none-any.whl (1.6 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.6/1.6 MB 125.9 MB/s eta 0:00:00[2025-08-20 10:32:32.100978] 

Downloading jinja2-3.1.5-py3-none-any.whl (134 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 134.6/134.6 kB 215.5 MB/s eta 0:00:00[2025-08-20 10:32:32.113992] 

Downloading jiter-0.9.0-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (352 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 352.4/352.4 kB 221.0 MB/s eta 0:00:00[2025-08-20 10:32:32.128554] 

Downloading joblib-1.5.1-py3-none-any.whl (307 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 307.7/307.7 kB 151.8 MB/s eta 0:00:00[2025-08-20 10:32:32.142282] 

Downloading json5-0.10.0-py3-none-any.whl (34 kB)

Downloading jsonpatch-1.33-py2.py3-none-any.whl (12 kB)

Downloading jsonpointer-3.0.0-py2.py3-none-any.whl (7.6 kB)

Downloading jsonschema-4.23.0-py3-none-any.whl (88 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 88.5/88.5 kB 125.8 MB/s eta 0:00:00[2025-08-20 10:32:32.182755] 

Downloading jsonschema_specifications-2024.10.1-py3-none-any.whl (18 kB)

Downloading jupyter-1.1.1-py2.py3-none-any.whl (2.7 kB)

Downloading jupyter_console-6.6.3-py3-none-any.whl (24 kB)

Downloading jupyter_events-0.11.0-py3-none-any.whl (19 kB)

Downloading jupyter_lsp-2.2.5-py3-none-any.whl (69 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 69.1/69.1 kB 132.2 MB/s eta 0:00:00

Downloading jupyter_client-8.6.3-py3-none-any.whl (106 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 106.1/106.1 kB 185.9 MB/s eta 0:00:00[2025-08-20 10:32:32.254108] 

Downloading jupyter_core-5.7.2-py3-none-any.whl (28 kB)

Downloading jupyter_server-2.15.0-py3-none-any.whl (385 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 385.8/385.8 kB 206.3 MB/s eta 0:00:00[2025-08-20 10:32:32.277763] 

Downloading jupyter_server_terminals-0.5.3-py3-none-any.whl (13 kB)

Downloading jupyterlab-4.3.4-py3-none-any.whl (11.7 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 11.7/11.7 MB 173.8 MB/s eta 0:00:00[2025-08-20 10:32:32.369985] 

Downloading jupyterlab_pygments-0.3.0-py3-none-any.whl (15 kB)

Downloading jupyterlab_server-2.27.3-py3-none-any.whl (59 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 59.7/59.7 kB 120.2 MB/s eta 0:00:00[2025-08-20 10:32:32.390132] 

Downloading jupyterlab_widgets-3.0.13-py3-none-any.whl (214 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 214.4/214.4 kB 180.9 MB/s eta 0:00:00[2025-08-20 10:32:32.403661] 

Downloading kiwisolver-1.4.8-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.5 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.5/1.5 MB 202.3 MB/s eta 0:00:00[2025-08-20 10:32:32.425024] 

Downloading kubernetes-33.1.0-py2.py3-none-any.whl (1.9 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.9/1.9 MB 208.4 MB/s eta 0:00:00[2025-08-20 10:32:32.446040] 

Downloading langchain-0.3.26-py3-none-any.whl (1.0 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.0/1.0 MB 134.6 MB/s eta 0:00:00[2025-08-20 10:32:32.469579] 

Downloading langchain_community-0.3.26-py3-none-any.whl (2.5 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.5/2.5 MB 142.4 MB/s eta 0:00:00[2025-08-20 10:32:32.506638] 

Downloading langchain_core-0.3.68-py3-none-any.whl (441 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 441.4/441.4 kB 159.6 MB/s eta 0:00:00[2025-08-20 10:32:32.526180] 

Downloading langchain_experimental-0.3.4-py3-none-any.whl (209 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 209.2/209.2 kB 197.6 MB/s eta 0:00:00[2025-08-20 10:32:32.540039] 

Downloading langchain_google_genai-2.1.5-py3-none-any.whl (44 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 44.8/44.8 kB 130.1 MB/s eta 0:00:00[2025-08-20 10:32:32.552816] 

Downloading langchain_google_vertexai-2.0.27-py3-none-any.whl (101 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 101.0/101.0 kB 121.7 MB/s eta 0:00:00[2025-08-20 10:32:32.568426] 

Downloading langchain_groq-0.3.2-py3-none-any.whl (15 kB)

Downloading langchain_text_splitters-0.3.8-py3-none-any.whl (32 kB)

Downloading langsmith-0.3.45-py3-none-any.whl (363 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 363.0/363.0 kB 123.0 MB/s eta 0:00:00[2025-08-20 10:32:32.605534] 

Downloading mando-0.7.1-py2.py3-none-any.whl (28 kB)

Downloading markdown_it_py-3.0.0-py3-none-any.whl (87 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 87.5/87.5 kB 126.9 MB/s eta 0:00:00[2025-08-20 10:32:32.630999] 

Downloading MarkupSafe-3.0.2-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (23 kB)

Downloading marshmallow-3.26.1-py3-none-any.whl (50 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 50.9/50.9 kB 98.6 MB/s eta 0:00:00[2025-08-20 10:32:32.653159] 

Downloading matplotlib-3.10.3-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (8.6 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8.6/8.6 MB 194.1 MB/s eta 0:00:00[2025-08-20 10:32:32.711701] 

Downloading matplotlib_inline-0.1.7-py3-none-any.whl (9.9 kB)

Downloading mccabe-0.7.0-py2.py3-none-any.whl (7.3 kB)

Downloading mdurl-0.1.2-py3-none-any.whl (10.0 kB)

Downloading mistune-3.1.0-py3-none-any.whl (53 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 53.7/53.7 kB 148.4 MB/s eta 0:00:00[2025-08-20 10:32:32.749335] 

Downloading mmh3-5.1.0-cp313-cp313-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (101 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 101.6/101.6 kB 191.1 MB/s eta 0:00:00[2025-08-20 10:32:32.761635] 

Downloading mpmath-1.3.0-py3-none-any.whl (536 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 536.2/536.2 kB 221.8 MB/s eta 0:00:00[2025-08-20 10:32:32.776069] 

Downloading multidict-6.3.2-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (248 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 248.5/248.5 kB 189.4 MB/s eta 0:00:00[2025-08-20 10:32:32.790030] 

Downloading mypy_extensions-1.1.0-py3-none-any.whl (5.0 kB)

Downloading narwhals-1.43.1-py3-none-any.whl (362 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 362.7/362.7 kB 205.9 MB/s eta 0:00:00[2025-08-20 10:32:32.810522] 

Downloading nbclient-0.10.2-py3-none-any.whl (25 kB)

Downloading nbconvert-7.16.5-py3-none-any.whl (258 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 258.1/258.1 kB 130.0 MB/s eta 0:00:00[2025-08-20 10:32:32.836944] 

Downloading nbformat-5.10.4-py3-none-any.whl (78 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 78.5/78.5 kB 129.0 MB/s eta 0:00:00[2025-08-20 10:32:32.849500] 

Downloading nest_asyncio-1.6.0-py3-none-any.whl (5.2 kB)

Downloading networkx-3.5-py3-none-any.whl (2.0 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.0/2.0 MB 136.5 MB/s eta 0:00:00[2025-08-20 10:32:32.886953] 

Downloading notebook-7.3.2-py3-none-any.whl (13.2 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 13.2/13.2 MB 137.0 MB/s eta 0:00:00[2025-08-20 10:32:33.003555] 

Downloading notebook_shim-0.2.4-py3-none-any.whl (13 kB)

Downloading numexpr-2.11.0-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (406 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 406.6/406.6 kB 144.4 MB/s eta 0:00:00[2025-08-20 10:32:33.031111] 

Downloading numpy-2.2.2-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (16.1 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 16.1/16.1 MB 141.4 MB/s eta 0:00:00[2025-08-20 10:32:33.162803] 

Downloading oauthlib-3.3.1-py3-none-any.whl (160 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 160.1/160.1 kB 139.4 MB/s eta 0:00:00[2025-08-20 10:32:33.176806] 

Downloading onnxruntime-1.22.1-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (16.5 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 16.5/16.5 MB 181.7 MB/s eta 0:00:00[2025-08-20 10:32:33.280786] 

Downloading openai-0.28.0-py3-none-any.whl (76 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 76.5/76.5 kB 198.2 MB/s eta 0:00:00[2025-08-20 10:32:33.292988] 

Downloading openpyxl-3.1.5-py2.py3-none-any.whl (250 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 250.9/250.9 kB 218.9 MB/s eta 0:00:00[2025-08-20 10:32:33.305079] 

Downloading opentelemetry_api-1.35.0-py3-none-any.whl (65 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 65.6/65.6 kB 130.9 MB/s eta 0:00:00[2025-08-20 10:32:33.317945] 

Downloading opentelemetry_exporter_otlp_proto_common-1.35.0-py3-none-any.whl (18 kB)

Downloading opentelemetry_exporter_otlp_proto_grpc-1.35.0-py3-none-any.whl (18 kB)

Downloading opentelemetry_proto-1.35.0-py3-none-any.whl (72 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 72.5/72.5 kB 115.5 MB/s eta 0:00:00[2025-08-20 10:32:33.353248] 

Downloading opentelemetry_sdk-1.35.0-py3-none-any.whl (119 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 119.4/119.4 kB 122.2 MB/s eta 0:00:00[2025-08-20 10:32:33.367932] 

Downloading opentelemetry_semantic_conventions-0.56b0-py3-none-any.whl (201 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 201.6/201.6 kB 140.7 MB/s eta 0:00:00[2025-08-20 10:32:33.385483] 

Downloading oracledb-3.1.0-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.manylinux_2_28_x86_64.whl (2.8 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.8/2.8 MB 130.8 MB/s eta 0:00:00[2025-08-20 10:32:33.424385] 

Downloading orjson-3.10.18-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (133 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 133.1/133.1 kB 138.1 MB/s eta 0:00:00[2025-08-20 10:32:33.439794] 

Downloading overrides-7.7.0-py3-none-any.whl (17 kB)

Downloading packaging-24.2-py3-none-any.whl (65 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 65.5/65.5 kB 137.5 MB/s eta 0:00:00[2025-08-20 10:32:33.465182] 

Downloading pandas-2.3.1-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (12.1 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 12.1/12.1 MB 117.8 MB/s eta 0:00:00[2025-08-20 10:32:33.603796] 

Downloading pandas_gbq-0.29.2-py3-none-any.whl (40 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 41.0/41.0 kB 127.0 MB/s eta 0:00:00[2025-08-20 10:32:33.616705] 

Downloading pandocfilters-1.5.1-py2.py3-none-any.whl (8.7 kB)

Downloading parso-0.8.4-py2.py3-none-any.whl (103 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 103.7/103.7 kB 119.8 MB/s eta 0:00:00[2025-08-20 10:32:33.644145] 

Downloading pathspec-0.12.1-py3-none-any.whl (31 kB)

Downloading pillow-11.2.1-cp313-cp313-manylinux_2_28_x86_64.whl (4.6 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.6/4.6 MB 73.6 MB/s eta 0:00:00[2025-08-20 10:32:33.744261] 

Downloading pinecone_client-6.0.0-py3-none-any.whl (6.7 kB)

Downloading pinecone_plugin_interface-0.0.7-py3-none-any.whl (6.2 kB)

Downloading platformdirs-4.3.6-py3-none-any.whl (18 kB)

Downloading plotly-6.2.0-py3-none-any.whl (9.6 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 9.6/9.6 MB 118.6 MB/s eta 0:00:00[2025-08-20 10:32:33.882483] 

Downloading posthog-5.4.0-py3-none-any.whl (105 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 105.4/105.4 kB 109.0 MB/s eta 0:00:00[2025-08-20 10:32:33.896346] 

Downloading prometheus_client-0.21.1-py3-none-any.whl (54 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 54.7/54.7 kB 102.6 MB/s eta 0:00:00[2025-08-20 10:32:33.909446] 

Downloading prompt_toolkit-3.0.50-py3-none-any.whl (387 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 387.8/387.8 kB 137.3 MB/s eta 0:00:00[2025-08-20 10:32:33.925461] 

Downloading propcache-0.3.1-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (228 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 228.2/228.2 kB 152.6 MB/s eta 0:00:00[2025-08-20 10:32:33.941801] 

Downloading proto_plus-1.26.1-py3-none-any.whl (50 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 50.2/50.2 kB 119.0 MB/s eta 0:00:00[2025-08-20 10:32:33.955198] 

Downloading protobuf-6.31.1-cp39-abi3-manylinux2014_x86_64.whl (321 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 321.1/321.1 kB 130.9 MB/s eta 0:00:00[2025-08-20 10:32:33.972701] 

Downloading psutil-6.1.1-cp36-abi3-manylinux_2_12_x86_64.manylinux2010_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (287 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 287.5/287.5 kB 130.1 MB/s eta 0:00:00

Downloading pure_eval-0.2.3-py3-none-any.whl (11 kB)

Downloading pyarrow-19.0.1-cp313-cp313-manylinux_2_28_x86_64.whl (42.1 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 42.1/42.1 MB 183.4 MB/s eta 0:00:00[2025-08-20 10:32:34.292666] 

Downloading pyasn1-0.6.1-py3-none-any.whl (83 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 83.1/83.1 kB 197.7 MB/s eta 0:00:00[2025-08-20 10:32:34.303864] 

Downloading pyasn1_modules-0.4.2-py3-none-any.whl (181 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 181.3/181.3 kB 207.0 MB/s eta 0:00:00[2025-08-20 10:32:34.315538] 

Downloading pybase64-1.4.1-cp313-cp313-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (71 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 71.4/71.4 kB 205.5 MB/s eta 0:00:00[2025-08-20 10:32:34.328654] 

Downloading pycparser-2.22-py3-none-any.whl (117 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 117.6/117.6 kB 139.4 MB/s eta 0:00:00[2025-08-20 10:32:34.340500] 

Downloading pydantic-2.11.2-py3-none-any.whl (443 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 443.3/443.3 kB 141.6 MB/s eta 0:00:00[2025-08-20 10:32:34.357366] 

Downloading pydantic_settings-2.10.0-py3-none-any.whl (45 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 45.2/45.2 kB 121.8 MB/s eta 0:00:00[2025-08-20 10:32:34.371476] 

Downloading pydantic_core-2.33.1-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (2.0 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.0/2.0 MB 132.5 MB/s eta 0:00:00[2025-08-20 10:32:34.406815] 

Downloading pydata_google_auth-1.9.1-py2.py3-none-any.whl (15 kB)

Downloading pydeck-0.9.1-py2.py3-none-any.whl (6.9 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 6.9/6.9 MB 127.6 MB/s eta 0:00:00[2025-08-20 10:32:34.484426] 

Downloading pygments-2.19.1-py3-none-any.whl (1.2 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.2/1.2 MB 148.3 MB/s eta 0:00:00[2025-08-20 10:32:34.503467] 

Downloading pylint-3.3.7-py3-none-any.whl (522 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 522.6/522.6 kB 128.5 MB/s eta 0:00:00[2025-08-20 10:32:34.521176] 

Downloading pyparsing-3.2.3-py3-none-any.whl (111 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 111.1/111.1 kB 127.2 MB/s eta 0:00:00[2025-08-20 10:32:34.534590] 

Downloading pypdf-5.8.0-py3-none-any.whl (309 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 309.7/309.7 kB 131.9 MB/s eta 0:00:00[2025-08-20 10:32:34.550255] 

Downloading pyproject_hooks-1.2.0-py3-none-any.whl (10 kB)

Downloading pyreadline3-3.5.4-py3-none-any.whl (83 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 83.2/83.2 kB 105.8 MB/s eta 0:00:00[2025-08-20 10:32:34.572510] 

Downloading python_dateutil-2.9.0.post0-py2.py3-none-any.whl (229 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 229.9/229.9 kB 129.4 MB/s eta 0:00:00[2025-08-20 10:32:34.586814] 

Downloading python_dotenv-1.1.0-py3-none-any.whl (20 kB)

Downloading python_json_logger-3.2.1-py3-none-any.whl (14 kB)

Downloading pytz-2024.2-py2.py3-none-any.whl (508 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 508.0/508.0 kB 224.7 MB/s eta 0:00:00[2025-08-20 10:32:34.617535] 

Downloading PyYAML-6.0.2-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (759 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 759.5/759.5 kB 223.0 MB/s eta 0:00:00[2025-08-20 10:32:34.632132] 

Downloading pyzmq-26.2.0-cp313-cp313-manylinux_2_28_x86_64.whl (860 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 860.3/860.3 kB 193.4 MB/s eta 0:00:00[2025-08-20 10:32:34.650968] 

Downloading radon-6.0.1-py2.py3-none-any.whl (52 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 52.8/52.8 kB 186.5 MB/s eta 0:00:00[2025-08-20 10:32:34.663263] 

Downloading referencing-0.36.1-py3-none-any.whl (26 kB)

Downloading regex-2024.11.6-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (796 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 796.9/796.9 kB 207.3 MB/s eta 0:00:00[2025-08-20 10:32:34.691708] 

Downloading requests-2.32.4-py3-none-any.whl (64 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 64.8/64.8 kB 148.4 MB/s eta 0:00:00[2025-08-20 10:32:34.704230] 

Downloading requests_oauthlib-2.0.0-py2.py3-none-any.whl (24 kB)

Downloading requests_toolbelt-1.0.0-py2.py3-none-any.whl (54 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 54.5/54.5 kB 100.7 MB/s eta 0:00:00[2025-08-20 10:32:34.727765] 

Downloading rfc3339_validator-0.1.4-py2.py3-none-any.whl (3.5 kB)

Downloading rfc3986_validator-0.1.1-py2.py3-none-any.whl (4.2 kB)

Downloading rich-14.0.0-py3-none-any.whl (243 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 243.2/243.2 kB 132.0 MB/s eta 0:00:00[2025-08-20 10:32:34.760783] 

Downloading rpds_py-0.22.3-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (385 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 385.2/385.2 kB 131.4 MB/s eta 0:00:00[2025-08-20 10:32:34.776981] 

Downloading rsa-4.9.1-py3-none-any.whl (34 kB)

Downloading safetensors-0.5.3-cp38-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (471 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 471.6/471.6 kB 144.5 MB/s eta 0:00:00[2025-08-20 10:32:34.803066] 

Downloading scikit_learn-1.7.1-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (9.4 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 9.4/9.4 MB 116.5 MB/s eta 0:00:00[2025-08-20 10:32:34.901200] 

Downloading scipy-1.16.0-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (35.1 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 35.1/35.1 MB 208.8 MB/s eta 0:00:00[2025-08-20 10:32:35.145981] 

Downloading seaborn-0.13.2-py3-none-any.whl (294 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 294.9/294.9 kB 224.4 MB/s eta 0:00:00[2025-08-20 10:32:35.158299] 

Downloading Send2Trash-1.8.3-py3-none-any.whl (18 kB)

Downloading sentence_transformers-5.0.0-py3-none-any.whl (470 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 470.2/470.2 kB 167.6 MB/s eta 0:00:00[2025-08-20 10:32:35.182641] 

Downloading setuptools-75.8.0-py3-none-any.whl (1.2 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.2/1.2 MB 156.4 MB/s eta 0:00:00[2025-08-20 10:32:35.203791] 

Downloading shapely-2.1.1-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (3.1 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.1/3.1 MB 199.0 MB/s eta 0:00:00[2025-08-20 10:32:35.235485] 

Downloading shellingham-1.5.4-py2.py3-none-any.whl (9.8 kB)

Downloading six-1.17.0-py2.py3-none-any.whl (11 kB)

Downloading smmap-5.0.2-py3-none-any.whl (24 kB)

Downloading sniffio-1.3.1-py3-none-any.whl (10 kB)

Downloading soupsieve-2.6-py3-none-any.whl (36 kB)

Downloading sqlalchemy-2.0.41-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (3.2 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.2/3.2 MB 189.0 MB/s eta 0:00:00[2025-08-20 10:32:35.307051] 

Downloading sqlparse-0.5.3-py3-none-any.whl (44 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 44.4/44.4 kB 204.2 MB/s eta 0:00:00[2025-08-20 10:32:35.317465] 

Downloading stack_data-0.6.3-py3-none-any.whl (24 kB)

Downloading streamlit-1.46.0-py3-none-any.whl (10.1 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 10.1/10.1 MB 214.0 MB/s eta 0:00:00[2025-08-20 10:32:35.389777] 

Downloading sympy-1.14.0-py3-none-any.whl (6.3 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 6.3/6.3 MB 215.5 MB/s eta 0:00:00[2025-08-20 10:32:35.430364] 

Downloading tabulate-0.9.0-py3-none-any.whl (35 kB)

Downloading tenacity-8.5.0-py3-none-any.whl (28 kB)

Downloading terminado-0.18.1-py3-none-any.whl (14 kB)

Downloading threadpoolctl-3.6.0-py3-none-any.whl (18 kB)

Downloading tinycss2-1.4.0-py3-none-any.whl (26 kB)

Downloading tokenizers-0.21.2-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (3.1 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.1/3.1 MB 195.4 MB/s eta 0:00:00[2025-08-20 10:32:35.504618] 

Downloading toml-0.10.2-py2.py3-none-any.whl (16 kB)

Downloading tomlkit-0.13.3-py3-none-any.whl (38 kB)

Downloading torch-2.7.1-cp313-cp313-manylinux_2_28_x86_64.whl (821.0 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 821.0/821.0 MB 176.5 MB/s eta 0:00:00[2025-08-20 10:32:41.181137] 

Downloading tornado-6.4.2-cp38-abi3-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (437 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 437.2/437.2 kB 108.0 MB/s eta 0:00:00[2025-08-20 10:32:41.197580] 

Downloading tqdm-4.67.1-py3-none-any.whl (78 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 78.5/78.5 kB 118.5 MB/s eta 0:00:00[2025-08-20 10:32:41.210472] 

Downloading traitlets-5.14.3-py3-none-any.whl (85 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 85.4/85.4 kB 117.4 MB/s eta 0:00:00[2025-08-20 10:32:41.223547] 

Downloading transformers-4.53.2-py3-none-any.whl (10.8 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 10.8/10.8 MB 106.3 MB/s eta 0:00:00[2025-08-20 10:32:41.341646] 

Downloading typer-0.16.0-py3-none-any.whl (46 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 46.3/46.3 kB 90.1 MB/s eta 0:00:00[2025-08-20 10:32:41.354251] 

Downloading types_python_dateutil-2.9.0.20241206-py3-none-any.whl (14 kB)

Downloading typing_inspect-0.9.0-py3-none-any.whl (8.8 kB)

Downloading typing_inspection-0.4.0-py3-none-any.whl (14 kB)

Downloading typing_extensions-4.12.2-py3-none-any.whl (37 kB)

Downloading tzdata-2025.1-py2.py3-none-any.whl (346 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 346.8/346.8 kB 112.8 MB/s eta 0:00:00[2025-08-20 10:32:41.408196] 

Downloading uri_template-1.3.0-py3-none-any.whl (11 kB)

Downloading urllib3-2.5.0-py3-none-any.whl (129 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 129.8/129.8 kB 124.5 MB/s eta 0:00:00[2025-08-20 10:32:41.432516] 

Downloading uvicorn-0.35.0-py3-none-any.whl (66 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 66.4/66.4 kB 116.8 MB/s eta 0:00:00[2025-08-20 10:32:41.446866] 

Downloading validators-0.35.0-py3-none-any.whl (44 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 44.7/44.7 kB 124.9 MB/s eta 0:00:00[2025-08-20 10:32:41.459404] 

Downloading watchdog-6.0.0-py3-none-manylinux2014_x86_64.whl (79 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 79.1/79.1 kB 127.3 MB/s eta 0:00:00[2025-08-20 10:32:41.471923] 

Downloading watchfiles-1.1.0-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (451 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 451.7/451.7 kB 228.6 MB/s eta 0:00:00[2025-08-20 10:32:41.488606] 

Downloading wcwidth-0.2.13-py2.py3-none-any.whl (34 kB)

Downloading webcolors-24.11.1-py3-none-any.whl (14 kB)

Downloading webencodings-0.5.1-py2.py3-none-any.whl (11 kB)

Downloading websocket_client-1.8.0-py3-none-any.whl (58 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 58.8/58.8 kB 144.5 MB/s eta 0:00:00[2025-08-20 10:32:41.524480] 

Downloading websockets-15.0.1-cp313-cp313-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (182 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 182.5/182.5 kB 147.2 MB/s eta 0:00:00[2025-08-20 10:32:41.537860] 

Downloading werkzeug-3.1.3-py3-none-any.whl (224 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 224.5/224.5 kB 156.9 MB/s eta 0:00:00[2025-08-20 10:32:41.550473] 

Downloading widgetsnbextension-4.0.13-py3-none-any.whl (2.3 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.3/2.3 MB 176.2 MB/s eta 0:00:00[2025-08-20 10:32:41.579399] 

Downloading yarl-1.19.0-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (347 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 347.6/347.6 kB 181.8 MB/s eta 0:00:00[2025-08-20 10:32:41.594892] 

Downloading zipp-3.23.0-py3-none-any.whl (10 kB)

Downloading zstandard-0.23.0-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (5.4 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5.4/5.4 MB 82.0 MB/s eta 0:00:00[2025-08-20 10:32:41.684030] 

Downloading nvidia_cublas_cu12-12.6.4.1-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (393.1 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 393.1/393.1 MB 121.0 MB/s eta 0:00:00[2025-08-20 10:32:44.553312] 

Downloading nvidia_cuda_cupti_cu12-12.6.80-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (8.9 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8.9/8.9 MB 120.0 MB/s eta 0:00:00[2025-08-20 10:32:44.640965] 

Downloading nvidia_cuda_nvrtc_cu12-12.6.77-py3-none-manylinux2014_x86_64.whl (23.7 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 23.7/23.7 MB 134.8 MB/s eta 0:00:00[2025-08-20 10:32:44.844347] 

Downloading nvidia_cuda_runtime_cu12-12.6.77-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (897 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 897.7/897.7 kB 179.8 MB/s eta 0:00:00[2025-08-20 10:32:44.861921] 

Downloading nvidia_cudnn_cu12-9.5.1.17-py3-none-manylinux_2_28_x86_64.whl (571.0 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 571.0/571.0 MB 125.5 MB/s eta 0:00:00[2025-08-20 10:32:48.651637] 

Downloading nvidia_cufft_cu12-11.3.0.4-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (200.2 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 200.2/200.2 MB 179.4 MB/s eta 0:00:00[2025-08-20 10:32:50.150992] 

Downloading nvidia_cufile_cu12-1.11.1.6-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (1.1 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.1/1.1 MB 153.6 MB/s eta 0:00:00[2025-08-20 10:32:50.171504] 

Downloading nvidia_curand_cu12-10.3.7.77-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (56.3 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 56.3/56.3 MB 199.0 MB/s eta 0:00:00[2025-08-20 10:32:50.525408] 

Downloading nvidia_cusolver_cu12-11.7.1.2-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (158.2 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 158.2/158.2 MB 111.9 MB/s eta 0:00:00[2025-08-20 10:32:51.688545] 

Downloading nvidia_cusparse_cu12-12.5.4.2-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (216.6 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 216.6/216.6 MB 149.3 MB/s eta 0:00:00[2025-08-20 10:32:53.143995] 

Downloading nvidia_cusparselt_cu12-0.6.3-py3-none-manylinux2014_x86_64.whl (156.8 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 156.8/156.8 MB 184.6 MB/s eta 0:00:00[2025-08-20 10:32:54.410158] 

Downloading nvidia_nccl_cu12-2.26.2-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (201.3 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 201.3/201.3 MB 126.1 MB/s eta 0:00:00[2025-08-20 10:32:55.831650] 

Downloading nvidia_nvjitlink_cu12-12.6.85-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl (19.7 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 19.7/19.7 MB 175.8 MB/s eta 0:00:00[2025-08-20 10:32:56.000233] 

Downloading nvidia_nvtx_cu12-12.6.77-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (89 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 89.3/89.3 kB 117.2 MB/s eta 0:00:00[2025-08-20 10:32:56.012468] 

Downloading triton-3.3.1-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (155.7 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 155.7/155.7 MB 88.6 MB/s eta 0:00:00[2025-08-20 10:32:57.234834] 

Downloading hf_xet-1.1.8-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (3.2 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.2/3.2 MB 169.0 MB/s eta 0:00:00

Downloading pexpect-4.9.0-py2.py3-none-any.whl (63 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 63.8/63.8 kB 185.9 MB/s eta 0:00:00[2025-08-20 10:32:57.283707] 

Downloading ptyprocess-0.7.0-py2.py3-none-any.whl (13 kB)

Downloading uvloop-0.21.0-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (4.7 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.7/4.7 MB 163.1 MB/s eta 0:00:00[2025-08-20 10:32:57.337223] 

Building wheels for collected packages: PyPika, pywinpty, sqlalchemy_bigquery

  Building wheel for PyPika (pyproject.toml): started

  Building wheel for PyPika (pyproject.toml): finished with status 'done'

  Created wheel for PyPika: filename=pypika-0.48.9-py2.py3-none-any.whl size=53803 sha256=2b365b3575709056ce6e49104653d6fdd755366751a81de6937022b37c52f1e6

  Stored in directory: /tmp/pip-ephem-wheel-cache-_om7rxb1/wheels/b4/f8/a5/28e9c1524d320f4b8eefdce0e487b5c2e128dbf2ed1bb4a60b

  Building wheel for pywinpty (pyproject.toml): started

  Building wheel for pywinpty (pyproject.toml): finished with status 'error'

  error: subprocess-exited-with-error

  

  × Building wheel for pywinpty (pyproject.toml) did not run successfully.

  │ exit code: 1

  ╰─> [119 lines of output]

      Running `maturin pep517 build-wheel -i /home/adminuser/venv/bin/python --compatibility off`

      Python reports SOABI: cpython-313-x86_64-linux-gnu

      Computed rustc target triple: x86_64-unknown-linux-gnu

      Installation directory: /home/adminuser/.cache/puccinialin

      Rustup already downloaded

      Installing rust to /home/adminuser/.cache/puccinialin/rustup

      warn: It looks like you have an existing rustup settings file at:

      warn: /home/adminuser/.rustup/settings.toml

      warn: Rustup will install the default toolchain as specified in the settings file,

      warn: instead of the one inferred from the default host triple.

      info: profile set to 'minimal'

      info: default host triple is x86_64-unknown-linux-gnu

      warn: Updating existing toolchain, profile choice will be ignored

      info: syncing channel updates for 'stable-x86_64-unknown-linux-gnu'

      info: default toolchain set to 'stable-x86_64-unknown-linux-gnu'

      Checking if cargo is installed

      cargo 1.89.0 (c24e10642 2025-06-23)

      ⚠️  Warning: `project.version` field is required in pyproject.toml unless it is present in the `project.dynamic` list

      📦 Including license file `LICENSE.txt`

      🍹 Building a mixed python/rust project

      🔗 Found pyo3 bindings

      🐍 Found CPython 3.13 at /home/adminuser/venv/bin/python

         Compiling proc-macro2 v1.0.88

         Compiling windows_x86_64_gnu v0.52.6

         Compiling unicode-ident v1.0.13

         Compiling target-lexicon v0.12.16

         Compiling autocfg v1.4.0

         Compiling once_cell v1.20.2

         Compiling rustix v0.38.37

         Compiling bitflags v2.6.0

         Compiling linux-raw-sys v0.4.14

         Compiling libc v0.2.160

         Compiling either v1.13.0

         Compiling home v0.5.9

         Compiling heck v0.5.0

         Compiling cfg-if v1.0.0

         Compiling unindent v0.2.3

         Compiling indoc v2.0.5

         Compiling windows-targets v0.52.6

         Compiling num-traits v0.2.19

         Compiling memoffset v0.9.1

         Compiling windows-result v0.2.0

         Compiling windows-strings v0.1.0

         Compiling pyo3-build-config v0.22.5

         Compiling quote v1.0.37

         Compiling syn v2.0.79

         Compiling pyo3-macros-backend v0.22.5

         Compiling pyo3-ffi v0.22.5

         Compiling pyo3 v0.22.5

         Compiling which v6.0.3

         Compiling windows-implement v0.58.0

         Compiling windows-interface v0.58.0

         Compiling enum-primitive-derive v0.3.0

         Compiling windows-core v0.58.0

         Compiling windows v0.58.0

         Compiling winpty-rs v0.4.0

      error[E0433]: failed to resolve: could not find `Win32` in `windows`

       --> /home/adminuser/.cache/puccinialin/cargo/registry/src/index.crates.io-1949cf8c6b5b557f/winpty-rs-0.4.0/src/pty/base.rs:3:14

        |

      3 | use windows::Win32::Foundation::{CloseHandle, HANDLE, STATUS_PENDING, S_OK, WAIT_FAILED, WAIT_OBJECT_0, WAIT_TIMEOUT};

        |              ^^^^^ could not find `Win32` in `windows`

      

      error[E0433]: failed to resolve: could not find `Win32` in `windows`

       --> /home/adminuser/.cache/puccinialin/cargo/registry/src/index.crates.io-1949cf8c6b5b557f/winpty-rs-0.4.0/src/pty/base.rs:4:14

        |

      4 | use windows::Win32::Storage::FileSystem::{GetFileSizeEx, ReadFile, WriteFile};

        |              ^^^^^ could not find `Win32` in `windows`

      

      error[E0433]: failed to resolve: could not find `Win32` in `windows`

       --> /home/adminuser/.cache/puccinialin/cargo/registry/src/index.crates.io-1949cf8c6b5b557f/winpty-rs-0.4.0/src/pty/base.rs:5:14

        |

      5 | use windows::Win32::System::Pipes::PeekNamedPipe;

        |              ^^^^^ could not find `Win32` in `windows`

      

      error[E0433]: failed to resolve: could not find `Win32` in `windows`

       --> /home/adminuser/.cache/puccinialin/cargo/registry/src/index.crates.io-1949cf8c6b5b557f/winpty-rs-0.4.0/src/pty/base.rs:6:14

        |

      6 | use windows::Win32::System::IO::CancelIoEx;

        |              ^^^^^ could not find `Win32` in `windows`

      

      error[E0433]: failed to resolve: could not find `Win32` in `windows`

       --> /home/adminuser/.cache/puccinialin/cargo/registry/src/index.crates.io-1949cf8c6b5b557f/winpty-rs-0.4.0/src/pty/base.rs:7:14

        |

      7 | use windows::Win32::System::Threading::{GetExitCodeProcess, GetProcessId, WaitForSingleObject};

        |              ^^^^^ could not find `Win32` in `windows`

      

      error[E0433]: failed to resolve: could not find `Win32` in `windows`

       --> /home/adminuser/.cache/puccinialin/cargo/registry/src/index.crates.io-1949cf8c6b5b557f/winpty-rs-0.4.0/src/pty/base.rs:8:14

        |

      8 | use windows::Win32::Globalization::{MultiByteToWideChar, WideCharToMultiByte, CP_UTF8, MULTI_BYTE_TO_WIDE_CHAR_FLAGS};

        |              ^^^^^ could not find `Win32` in `windows`

      

      error[E0433]: failed to resolve: could not find `Win32` in `windows`

        --> /home/adminuser/.cache/puccinialin/cargo/registry/src/index.crates.io-1949cf8c6b5b557f/winpty-rs-0.4.0/src/pty/base.rs:10:14

         |

      10 | use windows::Win32::System::Threading::INFINITE;

         |              ^^^^^ could not find `Win32` in `windows`

      

      error[E0432]: unresolved import `windows::core`

       --> /home/adminuser/.cache/puccinialin/cargo/registry/src/index.crates.io-1949cf8c6b5b557f/winpty-rs-0.4.0/src/pty/base.rs:9:14

        |

      9 | use windows::core::{HRESULT, Error, PCSTR};

        |              ^^^^ could not find `core` in `windows`

      

      error: the `Self` constructor can only be used with tuple or unit structs

        --> /home/adminuser/.cache/puccinialin/cargo/registry/src/index.crates.io-1949cf8c6b5b557f/winpty-rs-0.4.0/src/pty/base.rs:70:9

         |

      70 |         Self(value.0)

         |         ^^^^^^^^^^^^^

      

      Some errors have detailed explanations: E0432, E0433.

      For more information about an error, try `rustc --explain E0432`.

      error: could not compile `winpty-rs` (lib) due to 9 previous errors

      warning: build failed, waiting for other jobs to finish...

      💥 maturin failed

        Caused by: Failed to build a native library through cargo

        Caused by: Cargo build finished with "exit status: 101": `env -u CARGO PYO3_ENVIRONMENT_SIGNATURE="cpython-3.13-64bit" PYO3_PYTHON="/home/adminuser/venv/bin/python" PYTHON_SYS_EXECUTABLE="/home/adminuser/venv/bin/python" "cargo" "rustc" "--message-format" "json-render-diagnostics" "--manifest-path" "/tmp/pip-install-ic3engfr/pywinpty_899eef6758444f8abe67174c4b01e877/Cargo.toml" "--release" "--lib"`

      Rust not found, installing into a temporary directory

      Error: command ['maturin', 'pep517', 'build-wheel', '-i', '/home/adminuser/venv/bin/python', '--compatibility', 'off'] returned non-zero exit status 1

      [end of output]

  

  note: This error originates from a subprocess, and is likely not a problem with pip.

  ERROR: Failed building wheel for pywinpty

  Building wheel for sqlalchemy_bigquery (pyproject.toml): started

  Building wheel for sqlalchemy_bigquery (pyproject.toml): finished with status 'done'

  Created wheel for sqlalchemy_bigquery: filename=sqlalchemy_bigquery-0.0.7-py3-none-any.whl size=4620 sha256=efcb271b33a86c67ccbf7387a3ffe8affc831dd95f42f958098043e7bc47a84d

  Stored in directory: /tmp/pip-ephem-wheel-cache-_om7rxb1/wheels/18/f8/3a/e96ae18838495cc30ae741717b00fbcade0d081dd5d2a68dd2

Successfully built PyPika sqlalchemy_bigquery

Failed to build pywinpty

ERROR: Could not build wheels for pywinpty, which is required to install pyproject.toml-based projects


[notice] A new release of pip is available: 24.0 -> 25.2

[notice] To update, run: pip install --upgrade pip

Checking if Streamlit is installed

Installing rich for an improved exception logging

Using standard pip install.

Collecting rich>=10.14.0

  Downloading rich-14.1.0-py3-none-any.whl.metadata (18 kB)

Collecting markdown-it-py>=2.2.0 (from rich>=10.14.0)

  Downloading markdown_it_py-4.0.0-py3-none-any.whl.metadata (7.3 kB)

Collecting pygments<3.0.0,>=2.13.0 (from rich>=10.14.0)

  Downloading pygments-2.19.2-py3-none-any.whl.metadata (2.5 kB)

Collecting mdurl~=0.1 (from markdown-it-py>=2.2.0->rich>=10.14.0)

  Downloading mdurl-0.1.2-py3-none-any.whl.metadata (1.6 kB)

Downloading rich-14.1.0-py3-none-any.whl (243 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 243.4/243.4 kB 14.1 MB/s eta 0:00:00[2025-08-20 10:33:31.192341] 

Downloading markdown_it_py-4.0.0-py3-none-any.whl (87 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 87.3/87.3 kB 117.7 MB/s eta 0:00:00[2025-08-20 10:33:31.206824] 

Downloading pygments-2.19.2-py3-none-any.whl (1.2 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.2/1.2 MB 86.5 MB/s eta 0:00:00[2025-08-20 10:33:31.235324] 

Downloading mdurl-0.1.2-py3-none-any.whl (10.0 kB)

Installing collected packages: pygments, mdurl, markdown-it-py, rich

  Attempting uninstall: pygments

    Found existing installation: Pygments 2.19.2

    Uninstalling Pygments-2.19.2:

      Successfully uninstalled Pygments-2.19.2

  Attempting uninstall: mdurl

    Found existing installation: mdurl 0.1.2

    Uninstalling mdurl-0.1.2:

      Successfully uninstalled mdurl-0.1.2

  Attempting uninstall: markdown-it-py

    Found existing installation: markdown-it-py 4.0.0

    Uninstalling markdown-it-py-4.0.0:

      Successfully uninstalled markdown-it-py-4.0.0

  Attempting uninstall: rich

    Found existing installation: rich 14.1.0

    Uninstalling rich-14.1.0:

      Successfully uninstalled rich-14.1.0

Successfully installed markdown-it-py-4.0.0 mdurl-0.1.2 pygments-2.19.2 rich-14.1.0


[notice] A new release of pip is available: 24.0 -> 25.2

[notice] To update, run: pip install --upgrade pip


────────────────────────────────────────────────────────────────────────────────────────


[10:33:33] ❗️ installer returned a non-zero exit code

[10:33:33] ❗️ Error during processing dependencies! Please fix the error and push an update, or try restarting the app.