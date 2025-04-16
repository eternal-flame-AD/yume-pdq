# yume-pdq security policy

Yume-PDQ is committed to providing an accurate, low-latency, high-throughput and secure solution for PDQ hashing and matching.
As is documented in our engineering controls, we take security very seriously.

## Supported versions

For v0.x.x, only the latest version is supported, and the latest version is only supported until v1 is released.

For all minor releases since v1, all latest patch version and all versions that are less than 3 months old are supported.

However all versions less than 6 months old are eligible for a critical security update.

## Reporting a vulnerability

If you have found a security vulnerability, please report it to us responsibly. We will maintain confidentiality and are committed to working with you to resolve the issue as quickly and fully as possible.

You can either report it formally through GitHub private advisory (click on the `security` tab above and click `new advisory`) or send us a confidential email at [security@yumechi.jp](mailto:security@yumechi.jp) so we can reproduce your report. 
If you wish to author the patch, please indicate that in your report. We thank you for your contribution.


We aim to acknowledge and produce a preliminary response to all vulnerability reports within 5 business days. If you did not receive a response within 5 business days, please feel free to follow up using the alternative method you originally used to contact us. 

As a last resort, you may also contact the owner on fediverse at [@yume@mi.yumechi.jp](https://mi.yumechi.jp/@yume). However, please refrain from sharing any sensitive or confidential information through this channel.

Regardless of the method, you may choose to be credited or remain anonymous. You can indicate your preference by either using the GitHub advisory credit system or a clear statement in your report ("I wish to be credited" or "I wish to remain anonymous").

If no response is received, we will withhold your name from the advisory but we will make every effort to add you back should you later request it.

## Scoping 

These are considered security vulnerabilities:

- **Denial of service**: Infinite loops, significant performance degradation caused by crafted input, etc.
    - Large, risky stack allocations inside API functions that may trigger a stack overflow is considered a denial of service.
- **Information leak**: Unintended disclosure of memory, pointer addresses, files, or other resources.
- **Memory corruption**: Undefined behavior, buffer overflows, write-after-free, use-after-free, etc.
- **Bypass of security mitigations**: Circumventing protections like Control Flow Integrity (CFI), Address Space Layout Randomization (ASLR), or Non-Executable (NX) memory regions.
    - In particular, we consider a CFI bypass resulting in successful arbitrary code execution to be a vulnerability, even if the initial memory corruption is caused by a plausible caller error, such as an artificially induced heap or stack overflow.


These are considered bugs, and thus should be handled through the GitHub issue tracker:

- **Performance issues**:
    - Incorrect or suboptimal results related to statistical accuracy, recall, or algorithmic correctness. 
    - Security issue related to the use of `yume-pdq` output: if a numerical error in `yume-pdq` allowed certain adversarially-altered illegal images to be allowed through on your platform causing a security issue, that is considered a bug not a security vulnerability in `yume-pdq`.
    - Unstable performance that is not dependent on the input, or degradation that is not significant enough (<10x).
- **Documentation issues**:
    - Misleading API design or documentation that does not result in any security issues defined above.
        - For example, if a function accepts a temporary buffer not wrapped in `MaybeUninit` and exhibits undefined behavior when an uninitialized buffer is passed, this is considered a documentation improvement.
- **Test-specific vulnerabilities**:
    - Security vulnerabilities that are only reproducible in the examples or test suite and do not affect the core functionality of the project.

## Reward policy

This is a personal project and thus we can only offer a small, symbolic reward for a confirmed security vulnerability. The reward can be delivered as:
- A donation to your personal sponsorship address, including but not limited to GitHub Sponsors, ko-fi, Liberapay, OpenCollective, etc.
- A gift card to a reputable cloud provider of your choice, including but not limited to AWS, GCP, Azure, Vultr, Netcup.
- A donation to a charity with no political affiliation of your choice, including but not limited to NCMEC, EFF, FSF, Red Cross, Doctors Without Borders.

This reward is offered on a symbolic, discretionary, best effort basis. The reward will under no circumstances exceed 20 USD per reporter per month, and 100 USD accumulated per reporter.

## Disclosure policy

Due to the nature of low-level security vulnerability fixes, such as memory corruption bugs, the details of the fix often make it apparent where the issue lies and how to exploit it. To minimize the risk of malicious exploitation, we adopt a fast disclosure policy.
 
We will make every effort to disclose a security advisory within 7 days after the fix was committed.

In exceptional cases, such as critical vulnerabilities requiring extensive fixes, the disclosure timeline may be extended to ensure a comprehensive resolution.