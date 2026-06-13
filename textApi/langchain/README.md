
# Text Summary API (CDK + Lambda + LangChain + Amazon Bedrock)

This CDK (Python) app deploys a Lambda function behind an API Gateway REST API
that summarizes text using **LangChain** on top of **Amazon Bedrock** (Nova Lite).

`POST /text?points=<n>` with a JSON body `{ "text": "..." }` returns
`{ "summary": "..." }` containing an `n`-point summary.

## What changed (boto3 → LangChain)

The handler in `services/summary.py` no longer hand-builds the Bedrock
request/response payloads with raw `boto3`. Instead it uses an LCEL chain:

```
ChatPromptTemplate  ->  ChatBedrockConverse (langchain-aws)  ->  StrOutputParser
```

* `langchain_aws.ChatBedrockConverse` talks to the model `us.amazon.nova-lite-v1:0`
  via the Bedrock Converse API (it uses the runtime's bundled `boto3` underneath).
* The chain is constructed once at module import so warm invocations reuse it.
* The public API contract (request/response shape, IAM, route) is unchanged.

## Packaging the LangChain dependency (no Docker required)

`langchain-aws` / `langchain-core` are **not** part of the Lambda runtime, so the
CDK stack (`text_summary_app/py_stack.py`) bundles them into the function asset.
The Lambda's runtime dependencies live in `services/requirements.txt`.

Bundling runs **locally on the host — no Docker needed**. The stack supplies a
`local` bundler that runs `pip install` directly, forcing Lambda-compatible
wheels with:

```
--platform manylinux2014_x86_64 --implementation cp --python-version 3.11 --only-binary=:all:
```

so the package is correct even when you build on macOS or Windows. A Docker
image is still configured as a **fallback** and is only used if local bundling
fails (e.g. a dependency without a matching wheel). If you prefer the Lambda to
run on Graviton/ARM, switch the function `architecture` to `ARM_64` and change
the platform to `manylinux2014_aarch64`.

The `cdk.json` file tells the CDK Toolkit how to execute your app.

This project is set up like a standard Python project.  The initialization
process also creates a virtualenv within this project, stored under the `.venv`
directory.  To create the virtualenv it assumes that there is a `python3`
(or `python` for Windows) executable in your path with access to the `venv`
package. If for any reason the automatic creation of the virtualenv fails,
you can create the virtualenv manually.

To manually create a virtualenv on MacOS and Linux:

```
$ python3 -m venv .venv
```

After the init process completes and the virtualenv is created, you can use the following
step to activate your virtualenv.

```
$ source .venv/bin/activate
```

If you are a Windows platform, you would activate the virtualenv like this:

```
% .venv\Scripts\activate.bat
```

Once the virtualenv is activated, you can install the required dependencies.

```
$ pip install -r requirements.txt
```

At this point you can now synthesize the CloudFormation template for this code.

```
$ cdk synth
```

To add additional **CDK** dependencies, add them to `requirements.txt` and rerun
`pip install -r requirements.txt`. To add additional **Lambda runtime**
dependencies (anything `services/summary.py` imports), add them to
`services/requirements.txt` so they get bundled into the function.

## Running the tests

```
$ pip install -r requirements-dev.txt
$ python -m pytest tests
```

The tests disable bundling (via the `aws:cdk:bundling-stacks` context) so they
synthesize the stack quickly without installing the Lambda dependencies.

## Useful commands

 * `cdk ls`          list all stacks in the app
 * `cdk synth`       emits the synthesized CloudFormation template
 * `cdk deploy`      deploy this stack to your default AWS account/region
 * `cdk diff`        compare deployed stack with current state
 * `cdk docs`        open CDK documentation

Enjoy!
