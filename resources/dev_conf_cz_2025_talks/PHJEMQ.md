# Automatic joining cloud VMs to FreeIPA identity domain

**Speakers:** Alexandra Nikandrova, André Boscatto
                    
**Track:** Security and Compliance
                    
**Room:** 31
                    
**Date & Time:** 2025-06-12 13:15:00
                    
**Duration:** 35 minutes
                    
## Abstract
                    
Cloud workloads need to comply with your organisation's security policies. Joining them to an identity management domain can help with that, and automatically joining them is even better. Learn how the Podengo project enables automatic and secure enrollment of VMs into a FreeIPA domain.

FreeIPA is an open source identity management solution providing authentication, access control, and other security features for Linux machines, to help organisations meet their security and compliance objectives. These objectives persist when running workloads on public clouds. But the typical workflow of using SSH keys to access the machine may struggle to meet them.

Enter Podengo. The Podengo service registers your FreeIPA deployment (which could be on-premises), authenticates cloud VMs, and facilitates an automatic and secure domain enrollment. This presentation will explain how the protocol works, what is required to use it, and how we use the Podengo service to provide the Domain Join feature in Red Hat Hybrid Cloud Console (and how you could use it in other settings).

After covering the fundamentals and current use cases, we will discuss some of the feature gaps (and how to close them), and how we could add support for more identity management solutions.

This presentation could be particularly useful for system and cloud administrators, infosec people, and the cryptography-curious.

PS: There will be recorded demos available due to time constraints!

References:

https://www.freeipa.org/ https://github.com/podengo-project
