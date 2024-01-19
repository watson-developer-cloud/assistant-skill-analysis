import os

WA_SECRETS = ["WA_CONFIG", "WA_CONFIG_ACTION"]

if __name__ == "__main__":
    raise ValueError(len(str(os.environ["WA_CONFIG"]).strip()))
    for secret in WA_SECRETS:
        entry = os.environ[secret]
        with open("./" + secret.lower() + ".txt", "w", encoding="utf-8") as f:
            f.writelines(val + "\n" for val in entry.split(","))
