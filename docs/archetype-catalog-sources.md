# Archetype catalog: sources and upkeep

What the 100 detailed archetypes in `captain_claw/instructions/archetypes.json` rest on, written 2026-11-02. Use this when you update one, and before relying on a dated fact in one.

**How far to trust it.** Each batch was researched against the web, but a shared search limit ran out partway through, and EUR-Lex returned empty pages to the fetcher. So each section below separates what was read at the source from what came from a search snippet, a law firm's summary or established knowledge. The instructions tell agents to confirm anything in flux on an official page before relying on it, and every regulated archetype assists a qualified person rather than replacing one. None of this has had a practitioner's review.

**Re-checked after drafting:** Regulation (EU) 2026/1744 (AI Omnibus), Regulation (EU) 2026/2202 (EU261), Directive (EU) 2026/470 (CSRD Omnibus) and ISO 9001:2026, against secondary sources only. The check corrected `trade-compliance` and `ecommerce-manager`: the EUR 3 parcel duty is Regulation (EU) 2026/382, and the EUR 2 handling fee's basis is still reported differently.

**Soonest to date:** the BIS Affiliates Rule suspension (ends 10 November 2026), UK identity verification transition (November 2026), EU handling fee detail (`trade-compliance`, `ecommerce-manager`), revised ESRS (reported in force 10 November 2026), Charlotin's citation count (`legal-research-memo`).

**Adding or changing an archetype.** `tests/test_flight_deck/test_archetype_catalog.py` holds the shape: sections in order with hard limits first, tools it names, colleagues it names, specific keywords, the crisis rule for wellbeing families. Add its icon and any new family to `flight-deck/src/components/layout/archetypeLook.ts`. Keywords decide whether a planner sees it (`shortlist_archetypes`), so make them phrases people really type.


---

## Batch: accounting-investment


### VAT, ViDA, OSS and IOSS, VIES (vat-analyst, ap-clerk)
- https://taxation-customs.ec.europa.eu/taxation/vat/vat-digital-age-vida_en : Commission ViDA timetable (April 2025, 1 Jan 2027, 1 Jul 2028 with deferral to 2030, 1 Jul 2030, 1 Jan 2035) and 2026 implementation steps.
- https://www.revenue.ie/en/vat/vida-vat-modernisation/vida-implementation-timeline.aspx : Irish Revenue timeline, mandatory intra-EU e-invoicing and reporting from July 2030.
- https://sede.agenciatributaria.gob.es/Sede/en_gb/iva/novedades-iva/novedades-normativa-2025/directiva-2025-516-consejo-11-2025.html : Spanish tax agency, near-real-time systems replace VIES data from 1 Jul 2030; OSS extension and mandatory reverse charge from 1 Jul 2028.
- https://vat-one-stop-shop.ec.europa.eu/one-stop-shop_en : Union, non-Union and import schemes, EUR 10,000 threshold, EUR 150 limit, quarterly and monthly filing.
- https://trade.ec.europa.eu/access-to-markets/en/news/eu-applies-eu3-customs-duty-item-low-value-e-commerce-consignments : EUR 3 per item duty from 1 Jul 2026 to 1 Jul 2028, Regulation (EU) 2026/382, applies whichever VAT regime is used.
- https://europa.eu/youreurope/business/finance-and-tax/vat/check-vat-number-vies/index_en.htm : meaning of invalid (not activated for intra-EU trade), retry when unavailable, keep records of validations.
- https://ec.europa.eu/taxation_customs/vies/#/help : consultation number needs requester's own VAT number, "---" means no disclosure, valid is one element of evidence only (read through search excerpt; the page does not render through a fetch).
- https://ec.europa.eu/taxation_customs/vies/rest-api/check-status and https://ec.europa.eu/taxation_customs/vies/rest-api/ms/IE/vat/6388047V : tested live with a public VAT number; both respond to a plain fetch; requestIdentifier is empty without a requester.
- https://einvoice.belgium.be/en : Belgian mandatory structured B2B e-invoices from 1 Jan 2026.
- https://www.gov.uk/vat-registration/when-to-register : UK registration threshold GBP 90,000.
- https://www.salestaxinstitute.com/resources/economic-nexus-state-guide : chart dated 1 Aug 2026, USD 100,000 common threshold, states that dropped the 200-transaction test.

### Late payment (ar-collections)
- https://www.europarl.europa.eu/legislative-train/theme-a-europe-fit-for-the-digital-age/file-late-payments-directive-revision : file status "Blocked" by Council, no withdrawal, updated 20 Sep 2026.
- https://oeil.europarl.europa.eu/oeil/en/procedure-file?reference=2023/0323(COD) : procedure awaiting Council first reading position, no withdrawal listed.
- https://latepayclaim.com/blog/eu-late-payment-regulation-2026 : claims the file was "withdrawn"; read and NOT followed, because both Parliament pages contradict it.
- https://www.gesetze-im-internet.de/bgb/__288.html : Germany, 9 points over base rate and EUR 40 flat fee in B2B.
- https://www.ecb.europa.eu/stats/policy_and_exchange_rates/key_ecb_interest_rates/html/index.en.html : main refinancing rate moves during the year (2.65% from 16 Sep 2026), which is why the text says look up the rate instead of stating it.
- https://www.gov.uk/late-commercial-payments-interest-debt-recovery/charging-interest-commercial-debt : UK 8% plus Bank of England base rate.
- https://www.gov.uk/late-commercial-payments-interest-debt-recovery/claim-debt-recovery-costs : UK fixed sums GBP 40, 70, 100 by debt size.
- https://www.gov.uk/late-commercial-payments-interest-debt-recovery : UK default terms and 30-day rule.
- https://www.legal500.com/intelligence/united-kingdom/corporate-commercial-law/commercial-payments-bill-reforming-uk-late-payment-legislation : Commercial Payments Bill at report stage per article dated 30 Sep 2026, 60-day cap, interest terms.

### Payment fraud and payee checks (ap-clerk)
- https://www.ic3.gov/CrimeInfo/BEC : verify changes by a secondary channel; contact the originating bank at once for recall or reversal; file a complaint.
- https://www.ic3.gov/PSA/2017/PSA170504 : use previously known numbers, not numbers in the request (search excerpt).
- https://www.ic3.gov/AnnualReport/Reports/2022_ic3report.pdf : spoofed legitimate business phone numbers used to confirm false bank details (search excerpt).
- https://www.commerzbank.com/ms/documents/en/vop-verification-of-payee-faq-en.pdf : Verification of Payee results and warnings (search excerpt).
- https://paytechlaw.com/en/verification-of-payee-vop-under-the-instant-payment-regulation/ : VoP under the Instant Payments Regulation (search excerpt).
- https://docs.yapily.com/pages/payments/payments-resources/verification-of-payee : business opt-out for bulk payments (secondary, search excerpt). The 9 Oct 2025 and July 2027 dates were consistent across the bank and vendor pages in that search; the EPC page returned 403 and the regulation text was not read.
- https://engage.hoganlovells.com/knowledgeservices/news/app-fraud-uk-psr-confirms-reduced-maximum-level-of-reimbursement : UK reimbursement cap GBP 85,000 (search excerpt).
- https://help.refundclub.co.uk/claim-amounts/business-fraud : micro-enterprise limit on business eligibility (secondary, search excerpt).

### Audit (audit-senior)
- https://www.iaasb.org/news-events/2026-10/iaasb-enews-october-2026 : ISA 240 (Revised) and ISA 570 (Revised 2024) effective December 2026; ISA 330, 500, 520 still proposals.
- https://www.iaasb.org/focus-areas/strengthening-risk-based-audit-framework-proposed-revisions-isa-330-isa-500-and-isa-520 : exposure drafts, comments due 15 Dec 2026.
- https://www.iaasb.org/iaasb-consultations/latest-our-projects : project list; no separate ISA 530 project shown.
- https://pcaobus.org/news-events/news-releases/news-release-detail/pcaob-postpones-effective-date-of-qc-1000-and-related-standards--rules--and-forms : QC 1000 moved to 15 Dec 2026 (search excerpt; direct fetch returned 403).
- https://www.wilmerhale.com/en/insights/blogs/keeping-current-disclosure-and-governance-developments/20250829-pcaob-delays-implementation-of-qc-1000-until-december-2026 : same, secondary.
- https://pcaobus.org/oversight/standards/implementation-resources-PCAOB-standards-rules/general-responsibilities-of-the-auditor-in-conducting-an-audit-(as-1000) : AS 1000 and the 14-day documentation phase-in (search excerpt).

### Close (month-end-close)
- https://www.frc.org.uk/library/standards-codes-policy/accounting-and-reporting/uk-accounting-standards/frs-102/ : FRS 102 September 2024 edition, Periodic Review 2024 principal effective date 1 Jan 2026.

### Modelling (financial-modeler)
- https://fast-standard.org/wp-content/uploads/2019/10/FAST-Standard-02c-July-2019.pdf : rule numbers and wording, read from the PDF text.
- https://www.fast-standard.org/the-fast-standard/ : 02c July 2019 is the download offered as current; four FAST principles.

### Venture terms (term-sheet-analyst)
- https://nvca.org/model-legal-documents/ : last-updated dates of each model document.
- https://www.foley.com/insights/publications/2025/10/breaking-down-the-nvca-what-founders-and-vcs-need-to-know/ : October 2025 SPA changes, outbound and data-security representations.
- https://www.ukprivatecapital.co.uk/resource/bvca-model-documents-for-early-stage-investments-key-changes-to-february-2025-edition.html : February 2025 changes; BVCA now operates as UK Private Capital; EIS and VCT preference wording.
- https://www.nortonrosefulbright.com/en/knowledge/publications/1946bb06/updates-to-the-model-documents-a-short-guide-to-the-new-bvca-documents : non-participating preference, no intermediate leaver, warranties.

### Fund reporting (lp-reporting)
- https://ilpa.org/industry-guidance/templates-standards-model-documents/reporting-template : Reporting Template v2.0 and adoption rules.
- https://ilpa.org/wp-content/uploads/2025/04/ILPA-Templates-FAQs-v1.1.pdf : timing, delivery days, with and without facility, NAV facilities, recallables, subsequent close interest, granular and gross-up; read from the PDF text.
- https://www.proskauer.com/insights/get-pdf/30087 : note dated 17 Aug 2026, Capital Call & Distribution Template from Q1 2027, Portfolio Company proposal from Q1 2028, side letter "as amended" risk.
- https://ilpa.org/industry-guidance/templates-standards-model-documents/due-diligence-questionnaire-and-diversity-metrics-template/ : DDQ 2.0 last updated 2021; no newer version listed.
- https://www.privateequityvaluation.com/latest-news/details/IPEV-Board-publishes-updated-International-Private-Equity-and-Venture-Capital-Valuation-Guidelines-2025 : published 11 Dec 2025.
- https://www.lincolninternational.com/perspectives/articles/qa-update-to-international-private-equity-and-venture-capital-valuation-guidelines-ipev/ : periods beginning on or after 1 Apr 2026; recent price not a default; calibration.
- https://maples.com/wp-content/uploads/2026/09/AIFMD-II-Key-Implementation-Developments-in-Luxembourg.pdf : transposition uneven after 16 Apr 2026; reporting obligations from 16 Apr 2027.
- https://www.ivp.in/resources/blogs/aifmd-ii-readiness-checklist-what-to-do-now-next-and-before-april-2027/ : Annex IV April 2027; ESMA standards expected H2 2026 (vendor).
- https://www.ey.com/en_lu/insights/financial-services/esma-sets-direction-for-the-future-of-aifmd-ii-and-ucits-vi-reporting : revised RTS and ITS still to come; ESMA final report 4 May 2026.

### Stated from settled professional knowledge, not re-read live
- EUR-Lex returned empty pages to the fetch tool, so article numbers in Directive 2006/112/EC, Regulation 282/2011, Directive 2011/7/EU and AIFMD Art. 3(2) were not re-read today. They are long-standing.
- ISA 530, 320, 230, 500 and PCAOB AS 2315, AS 1215 text: IFAC links returned 404 and PCAOB pages 403.
- Monetary unit sampling factors, price-volume-mix algebra, weighted-average anti-dilution formula, DCF, LBO and SaaS metric definitions, TVPI, DPI, RVPI definitions.
- Limitation periods (Germany 3 years, England and Wales 6 years); staged e-invoicing in Germany, France and Poland (dates deliberately left out, the text sends the agent to the national page).
- Thresholds such as the 10% flux test, 60-day stale item flag, reminder ladder days and the 75% terminal value flag are working heuristics, not rules from a source.

---

## Batch: compliance-ops


Checked 2026-11-02. Primary sources first where one could be read. "Search summary" means the fact came from a search result summary, and the page itself was not opened.

### Incident response, NIS2, DORA, CRA, GDPR (soc-analyst, iso-soc2-auditor)
- https://csrc.nist.gov/pubs/sp/800/61/r3/final : NIST SP 800-61 Rev. 3, April 2025, CSF 2.0 Community Profile, supersedes Rev. 2.
- https://attack.mitre.org/resources/updates/ : ATT&CK v20 current, from 13 October 2026.
- https://www.nis-2-directive.com/NIS_2_Directive_Article_23.html : text of NIS2 Art. 23(3) significance test and 23(4) 24 hour, 72 hour, one month clocks.
- https://digital-strategy.ec.europa.eu/en/policies/nis2-directive : NIS2 scope, medium and large entities in critical sectors, amendments proposed 20 January 2026.
- https://www.hunton.com/privacy-and-cybersecurity-law-blog/european-commission-refers-four-member-states-to-cjeu-over-nis2-transposition-delays : Ireland, Spain, France, Netherlands referred to the CJEU, 8 July 2026.
- https://www.european-datarooms.eu/compliance/nis2-transposition-tracker-2026 and https://www.nis-2-templates.com/transposition-tracker/ : trackers disagree on the count, so the instructions say check the country (search summary).
- https://advisera.com/cdr-2025-301/time-limits-for-the-initial-notification-and-for-the-intermediate-and-final-reports/ : DORA Delegated Regulation 2025/301 Art. 5 time limits, 4 hours, 24 hours, 72 hours, one month.
- https://www.esma.europa.eu/sites/default/files/2026-09/DORA_Incident_reporting_-_Operational_instructions.pdf : DORA reporting instructions exist and are current (search summary).
- https://digital-strategy.ec.europa.eu/en/policies/cra-reporting : CRA reporting from 11 September 2026, 24 hour and 72 hour clocks, ENISA Single Reporting Platform.
- https://dpoas.co.il/blog/digital-omnibus-council-mandate-gdpr/ : Council mandate 21 October 2026, 96 hours and high-risk only, not law, Parliament position outstanding, GDPR Art. 33 72 hours still applies.
- https://diesec.com/2026/10/digital-omnibus-incident-reporting-council-position/ : national entry points in the Council position (search summary).

### ISO 27001, SOC 2 (iso-soc2-auditor)
- https://sprinto.com/iso-27001/controls/ and https://www.brightdefense.com/news/iso-27001-what-changed-in-2026/ : 93 controls, 37, 8, 14, 34; 11 new controls; transition to 31 October 2025; Amendment 1:2024 climate text, no new controls (search summary; vendor sources).
- https://www.barradvisory.com/resource/whats-new-with-soc-2/ : 2022 revised points of focus, 2017 criteria unchanged, AICPA wording, five categories.
- https://linfordco.com/blog/soc-2-common-criteria-cs-old-tsps/ : Security is the only required category (search summary).

### Questionnaires and vendor risk (questionnaire-responder)
- https://cloudsecurityalliance.org/blog/2026/02/19/ccm-v4-1-transition-timeline : CCM and CAIQ v4.1 January 2026, STAR accepts both until December 2027, v4.0.x withdrawn January 2028.
- https://cloudsecurityalliance.org/artifacts/star-level-1-security-questionnaire-caiq-v4-1 : CAIQ v4.1 released 27 January 2026, yes/no format.
- https://mitratech.com/resource-hub/blog/sig-2026-key-updates-and-considerations/ : SIG 2026 additions (search summary; vendor source).
- https://www.servicenow.com/docs/r/Z61H6JNUJXKe8NNPQzK7MQ/_PAntfWX755MlsrKV8ez_Q : SIG Core 2026 and SIG Lite 2026 listed (search summary).
- https://complydog.com/blog/what-is-a-sig-questionnaire : SIG EV launched 17 March 2026 (search summary; secondary).

### EU AI Act (ai-act-officer)
- https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai : official timeline after the AI Omnibus, in force 27 July 2026, Annex III 2 December 2027, Annex I 2 August 2028, December 2026 ban.
- https://digital-strategy.ec.europa.eu/en/faqs/ai-literacy-questions-answers : Art. 4 as amended, no guaranteed level, no certificate, no AI officer, enforcement from 2 August 2026.
- https://usercentrics.com/knowledge-hub/eu-ai-act-high-risk-delay-article-50-transparency-consent/ : Regulation (EU) 2026/1744, Art. 50 itself not amended, 50(2) grace period, Art. 5 addition.
- https://www.joneswalker.com/en/insights/blogs/ai-law-blog/yes-august-2-still-matters-the-eu-approved-a-high-risk-ai-delay-but-most-trans.html : Art. 50 duties in detail, 50(2) grace, penalty tier, SME lower figure, guidelines still pending at that date.
- https://www.gibsondunn.com/eu-ai-act-omnibus-agreement-postponed-high-risk-deadlines-and-other-key-changes/ : content of the provisional agreement, cross-check on Art. 4 and sandboxes.

### CSRD, ESRS, green claims, GHG (esg-analyst, ecommerce-manager)
- https://www.lw.com/en/insights/eu-sustainability-omnibus-published-in-the-official-journal : CSRD and CSDDD thresholds, non-EU rule, transposition 19 March 2027, CSDDD compliance 26 July 2029.
- https://aigreentools.com/csrd-scope-after-omnibus/ : Directive (EU) 2026/470 article references, first-wave option, FY2027 start, listed SMEs out, Delegated Regulation 2026/1563 (secondary, marked confirm in the instructions).
- https://www.mayerbrown.com/en/insights/publications/2026/07/european-commission-adopts-revised-european-sustainability-reporting-standards : revised ESRS adopted 3 July 2026, datapoint cuts, FY2027 application, FY2026 options, VSME cap from 1 January 2027.
- https://www.akingump.com/print/v2/content/1136434/eu-finalizes-streamlined-esrs-and-new-voluntary-reporting-standard.pdf : OJ publication 21 September 2026, in force 10 November 2026 (search summary).
- https://www.lw.com/en/insights/eu-empowering-consumers-directive-new-rules-on-green-claims-apply-from-27-september-2026 : Directive 2024/825 bans and requirements, B2C scope, dates, infringement cases, old stock.
- https://cms.law/en/int/publication/the-green-claims-directive-a-legal-update-on-developments-evolving-enforcement-and-overview-of-local-regulations-and-judgments-in-selected-juri and https://www.climatepartner.com/en/knowledge/glossary/green-claims-directive : Green Claims Directive on hold since June 2025, formal status unclear (search summary).
- https://ghgprotocol.org/ghg-protocol-public-consultations : Scope 2 consultation closed 31 January 2026, Corporate Standard draft expected Q2 2027.

### Procurement rules (procurement-analyst)
- https://environment.ec.europa.eu/topics/forests/deforestation/regulation-deforestation-free-products_en : EUDR application dates 30 December 2026 and 30 June 2027.

### Customs and trade (trade-compliance, ecommerce-manager)
- https://taxation-customs.ec.europa.eu/customs/eu-customs-reform_en : Regulation (EU) 2026/2108 adopted 16 September 2026, EUR 3 duty dates, EUR 2 handling fee from 1 November 2026, Data Hub dates, EU Customs Authority.
- https://www.consilium.europa.eu/en/press/press-releases/2026/02/11/council-gives-final-green-light-to-new-customs-duty-rules-for-small-parcels/ : EUR 3 duty rules (search summary).
- https://www.wcoomd.org/en/topics/nomenclature/instrument-and-tools/hs-nomenclature-2028-edition.aspx : HS 2028 in force 1 January 2028, correlation tables published.
- https://www.wcoomd.org/en/media/newsroom/2026/january/hs-2028-amendments-adapting-the-harmonized-system-to-global-priorities-and-trade-evolution.aspx : 299 sets of amendments, accepted January 2026.
- https://taxation-customs.ec.europa.eu/online-services/online-services-and-databases-customs/european-binding-tariff-information-ebti_en : BTI generally valid three years, binding across the EU (search summary).
- https://www.dhl.com/discover/en-br/news-and-insights/next-incoterms-revision : Incoterms 2020 still operative, ICC 2026 workplan starts the next revision (search summary).
- https://taxation-customs.ec.europa.eu/carbon-border-adjustment-mechanism_en : CBAM definitive since 1 January 2026, six sectors.
- https://www.dehst.de/EN/Topics/CBAM/CBAM-definitive-regime-2026/CBAM-certificates/cbam-certificates_node.html : certificates on sale 1 February 2027, surrender 30 September 2027, holding rule, prices.
- https://www.arthurcox.com/knowledge/carbon-border-adjustment-mechanism-simplification/ and https://www.ey.com/en_gl/technical/tax-alerts/eu-adopts-cbam-omnibus-regulation : 50 tonne net mass threshold, electricity and hydrogen excluded, declarant rule.
- https://www.troutman.com/insights/supreme-court-strikes-down-ieepa-tariffs-trump-responds-with-section-122-global-surcharge/ and https://www.americanbar.org/groups/public_contract_law/resources/procurement-lawyer/2026-fall/ieepa-tariffs-struck-down-replacement-tariffs-expired/ : IEEPA ruling of 20 February 2026 and what followed (search summary).

### Quality (quality-engineer)
- https://www.logincident.com/glossary/quality-management/iso-9001-2026-transition/ : ISO 9001:2026 published 16 September 2026, transition to 30 September 2029, Global ACI sets it, 2015 certificates remain valid (secondary, marked confirm in the instructions).
- https://www.lrqa.com/en-us/latest-news/iso-9001-revision-update-publication-date-confirmed/ : publication date and three-year transition, from a certification body.
- https://reliamag.com/guides/how-to-perform-fmea/ and https://eprints.uklo.edu.mk/id/eprint/6566/ : AIAG-VDA Action Priority replacing RPN, seven steps, RPN example (search summary).

### Forecasting (supply-planner)
- https://otexts.com/fpp3/accuracy.html : limits of MAPE, MASE definition and the meaning of a value below 1.

### E-commerce law (ecommerce-manager)
- https://www.noerr.com/en/insights/impact-of-the-cjeu-ruling-on-discount-advertising-in-europe-what-it-means-for-businesses : C-330/23 holding, percentage and promotional wording, strike-through prices.
- https://eur-lex.europa.eu/eli/C/2024/6621/oj/eng and https://curia.europa.eu/jcms/upload/docs/application/pdf/2024-09/cp240152en.pdf : C-330/23 date and headline holding (search summary and PDF title only).
- https://www.vimm.be/en/blog/gpsr-article-19-product-information-webshop : GPSR Art. 19 listing fields, Art. 16 responsible person.
- https://webgate.ec.europa.eu/safety/consumers/consumers_safety_gate/obligationsForBusinesses/documents/GPSR-Presentation-website.pdf : Safety Business Gateway for accident notices (search summary).
- https://www.gtlaw.com/en/insights/2026/5/eu-consumer-law-new-withdrawal-button-requirements-for-online-contracts : Directive 2023/2673 withdrawal function from 19 June 2026 (search summary).
- https://environment.ec.europa.eu/topics/waste-and-recycling/packaging-waste_en : PPWR applies from 12 August 2026.

### Written from established knowledge, not re-checked live this session
The shared web search budget ran out before these could be looked up. They are long-standing or are written as "check":
- General Rules of Interpretation, Incoterms 2020 rule details, Late Payment Directive 2011/7/EU terms, Kraljic matrix, EU Forced Labour Regulation 2024/3015 date.
- GDPR Arts. 28, 33, 34, EDPB Guidelines 9/2022, NIS2 Arts. 20 and 21, DORA Arts. 4 and 30, ISO 27001 clause numbers and new control numbers.
- AI Act Arts. 5, 6(3), 25, 26, 27, 99, Commission guideline points on text sentiment and the one-third compute threshold.
- Regulation 833/2014 Arts. 8a and 12g, Regulation 2021/821 Art. 4, BIS 50% Affiliates Rule suspension (written as "check").
- ISO 9001:2015 clause 10.2, the 8D steps, SPC signal rules, Cpk conventions, %GRR bands, safety stock, reorder point and EOQ formulas, z values.
- UCPD review rules, UK DMCC Act, US FTC review rule, Art. 6a Member State options.

---

## Batch: edu-media-local


Checked 2026-11-02. "Read" means the page or PDF text was opened. "Snippet" means only the search result text was seen.

### Literature review
- https://pmc.ncbi.nlm.nih.gov/articles/PMC8005924/ : read. PRISMA 2020 has 27 items; wording of items 7, 8, 11, 14, 15, 24a.
- https://www.prisma-statement.org/prisma-2020 : read. Components: statement, checklist, abstract checklist, flow diagrams, explanation paper.
- https://eprints.whiterose.ac.uk/150579/1/bmj.l4898.full.pdf : read. RoB 2 five domains, judgment levels, overall rule, judged per result, assignment versus adhering.
- https://www.riskofbias.info/welcome/rob-2-0-tool/current-version-of-rob-2 : read. Current RoB 2 dated 22 August 2019.
- https://www.riskofbias.info/welcome/robins-i-v2 : read. ROBINS-I V2 posted 20 November 2025, still a draft.
- https://www.cochrane.org/events/version-2-robins-i-tool : snippet. ROBINS-I V2 released during 2025.
- https://www.cochrane.org/authors/handbooks-and-manuals/handbook/current : read. Handbook 6.5, chapters 8 and 25. Used in research; line cut from the final text for length.
- https://pmc.ncbi.nlm.nih.gov/articles/PMC12594113/ : read. 2025 joint AI position of Cochrane, Campbell, JBI and CEE: accountability, oversight, disclosure fields.

### Statistics
- https://www.sciencedaily.com/releases/2016/03/160307092305.htm : snippet. ASA statement principles and DOI 10.1080/00031305.2016.1154108.
- https://arxiv.org/pdf/1512.04922 : snippet. Johari et al.: continuous monitoring drives Type I error toward 100%; always-valid inference.
- https://www.lukasvermeer.nl/publications/papers/2019/07/25/diagnosing-sample-ratio-mismatch-in-online-controlled-experiments.html : read. Fabijan et al. 2019 exists, taxonomy of SRM types. Abstract only.
- https://en.wikipedia.org/wiki/Sample_ratio_mismatch : read. Chi-squared test is the detection method; failures of randomisation and instrumentation.
- https://metricgate.com/docs/sample-ratio-mismatch-test/ : snippet. Strict alpha of 0.001 or 0.0001 in practice. Commercial source; written as "commonly".

### Grants
- https://sciencebusiness.net/sites/default/files/inline-files/HORIZON-General-annexes-2026-2027_05_18_2026.pdf : read. Horizon 2026-2027 General Annexes: criteria, thresholds, 1.5 Impact weight, two-stage cut, page limits, blind evaluation, ties, gender equality plan, lump sum budget table. Copy hosted by Science Business; the official URL returned 404.
- https://www.gov.si/assets/ministrstva/MVZI/Znanost/Obzorje-Evropa/Dogodki/2026-Dogodek-Obzorje-Evropa/Simplification-in-Horizon-Europe.pdf : snippet. Excellence and Implementation criteria unchanged; Impact template change applies to 2026-2027 calls.
- https://www.ukro.ac.uk/news/summary-of-ec-webinar-on-novelties-in-the-2026-2027-horizon-europe-work-programme/ : snippet. Less information requested under Implementation. The page now redirects.
- https://cairibu.urology.wisc.edu/category/nih-guidance/ : snippet. Simplified review from due dates of 25 January 2025; three factors.
- https://research.osu.edu/nih-policy-update-annual-limit-applications-and-ai-use-grant-proposals : snippet. NOT-OD-25-132 AI and six-application summary.
- https://grad.uic.edu/news-stories/nih-limits-ai-use-caps-grant-apps/ : snippet. Six-application cap and AI restriction. NIH's own notice returned 403.
- https://osr.ucsf.edu/node/2516 : snippet. Unified Funding Strategy, no paylines from January 2026 Councils.
- https://faseb.org/journals-and-news/washington-update/nih-proposes-changes-to-peer-review-reporting-outcomes : read. NOT-OD-26-088: three categories proposed, comments closed 13 October 2026.

### Newsroom
- https://hapgood.us/2019/06/19/sift-the-four-moves/ : read. SIFT moves in Caulfield's words.
- https://c2paviewer.com/articles/how-to-verify-c2pa-content : snippet. C2PA shows origin, not truth; a misleading item can carry a valid signature; platforms strip manifests.
- https://www.computing.co.uk/news/4181442/bbc-news-marks-content-verified-counter-disinformation : snippet. Newsrooms using provenance marks as part of verification.
- https://eapil.org/2026/05/07/status-of-the-anti-slapp-directive-transpositions-at-the-7-may-2026-deadline/ : snippet. Directive deadline and uneven transposition.
- https://eapil.org/2026/07/16/commission-opens-infringement-proceedings-over-anti-slapp-directive-transposition/ : snippet. Infringement proceedings, July 2026. Used in research; detail cut from the final text for length.

### Translation
- https://themqm.org/the-mqm-typology/ : read. Seven MQM top-level error types; consistency with ISO 5060.
- https://arxiv.org/pdf/2104.14478 : snippet. MQM penalty convention: weights of minor 1, major 5, critical 10 and critical at 25 both appeared across the MT-evaluation papers this search returned; the summary did not pin each figure to one paper, so the text says "10 or 25" and "agree weights first".
- https://atc.org.uk/wp-content/uploads/ISO-Blog-18587-Machine-translation-post-editing.pdf : read. ISO 18587 revision expected October 2026; light and full post-editing retained.
- https://atc.org.uk/wp-content/uploads/ATC-ISO-Blog-17100-Translation-Services-Review.pdf : read. ISO 17100 revision and its open questions, including competence and post-editing.
- https://genorma.com/en/standards/pren-iso-17100-rev : snippet. Revision project approved 24 July 2025.
- https://www.m21global.com/en/blog/what-is-the-new-iso-17100-translation-standard : read. Check, revision (bilingual, mandatory), review, proofreading, final verification.
- https://en.wikipedia.org/wiki/ISO_17100 : read. Revision mandatory, review and proofreading optional.

### Video and podcast rights
- https://www.finchley.co.uk/finchley-learning/podcast-uk-licensing-and-music-what-you-need-to-know : snippet. PRS and MCPS podcast licence; separate sync licence for commercial or production music.
- https://www.prsformusic.com/c/podcasts-consultation : snippet. PRS podcasts and audiobooks scheme consultation.
- https://www.soundstripe.com/blogs/music-rights-for-podcasts : snippet. Composition and master, both needed; no length threshold; purchase is not a licence. Vendor source.
- https://www.pinsentmasons.com/out-law/guides/podcasting-the-legal-issues : read. Short samples need licences; masters negotiated with the label. Guide dated 2008, used only as support.

### Construction
- https://www.fenwickelliott.com/knowledge-hub/international-quarterlies/issue-25/changes-to-the-claims-provisions-in-the-2017-fidic-red-book/ : read. Sub-clause 20.2: 28 days, 84 days, discharge.
- https://howardkennedy.com/Latest/Article/FIDIC-2022-Reprints_10-Key-Areas-Of-Change-In-The-FIDIC-Red-Book-2017 : snippet. FIDIC 2022 reprints exist. Page body returned 403.
- https://www.thrings.com/blog/jct-stands-down-2026-contract-suite : read. JCT 2016 suite withdrawn March 2026; do not lift old amendments.
- https://prettys.co.uk/articles/jct-2024-editions-whats-new-and-whats-missing : snippet. JCT 2024 themes and the Minor Works gap on relevant events.
- https://gowlingwlg.com/en-ae/insights-resources/articles/2024/jct-publishes-minor-works-2024 : snippet. Minor Works 2024.
- https://canadabuys.canada.ca/documents/pub/att/2021/10/28/2305b22cf3ce2931a9ca532a8274cba4/appendix_e_-_annex_7.2_-_classification_system_cost_estimate_matrix.pdf : read. AACE 18R-97 (rev. 2011) class table, process industries. Government-hosted copy.
- https://buildxact.com/us/blog/what-is-the-process-of-bid-leveling/ : snippet. Bid levelling purpose and process.
- https://downtobid.com/blog/bid-leveling : snippet. Levelling sheet columns; plug numbers kept internal. Vendor source.

### Hospitality
- https://www.legislation.gov.uk/eur/2011/1169/annex/II : read. The 14 allergen groups. UK retained text; EUR-Lex returned an empty page.
- https://www.legislation.gov.uk/eur/2011/1169/article/44 : read. Article 44, mandatory for non-prepacked food; UK-amended wording, so the "national rules" phrasing rests on the EU original from memory.
- https://www.legislation.gov.uk/eur/2004/852/article/5 : read. Seven HACCP principles.
- https://www.FSAI.ie/getattachment/35d81414-f56c-4111-8911-98c62b3b1dba/OJ_C_2022_355_FULL_EN_TXT.pdf?lang=en-IE : snippet. Commission Notice 2022/C 355/01, flexibility for certain establishments.
- https://eur-lex.europa.eu/eli/reg/2021/382 : snippet. Regulation 2021/382: allergen management, redistribution, food safety culture.
- https://www.food.gov.uk/business-guidance/cooking-safely-in-your-business : snippet. 70C for 2 minutes or 75C for 30 seconds.
- https://www.food.gov.uk/business-guidance/chilling-food-correctly-in-your-business : snippet. 8C legal limit in England, Wales and Northern Ireland; fridges at 5C or below.
- https://assets.publishing.service.gov.uk/media/69c53a16b66ff902f454428e/sfbb-caterer-cooking-06-hot-holding_0.pdf : snippet. Hot holding at 63C, one 2-hour allowance.
- https://www.loadedhub.com/resources/menu-engineering-matrix-restaurant : read. Quadrants, actions, 4 to 8 week period, compare within course. Vendor source.
- https://publishingindia.com/archive/johar/application-of-kasavana-smith-menu-engineering-model-to-menu-of-a-resort-restaurant-a-case-study-approach : snippet. Kasavana and Smith model named in the literature; authors and 1982 date came from the same search's results.
- https://www.netsuite.co.uk/portal/uk/resource/articles/accounting/restaurant-food-costs.shtml : snippet. Food cost 28 to 35% in UK restaurants; no official benchmark. Vendor source.
- https://beancount.io/blog/2026/05/15/weekly-prime-cost-tracking-restaurants-food-beverage-labor-55-65-percent-benchmark-margin-leaks-operators-guide : snippet. Prime cost 55 to 65%. Vendor source.
- https://supy.io/blog/learn-restaurant-labor-cost-percentage : snippet. Labour 28 to 35%. Vendor source.
- https://www.shiftbase.com/blog/how-to-cut-labour-costs-in-restaurants-uk : snippet. Labour 25 to 30%. Vendor source.

### Field service
- https://europa.eu/youreurope/citizens/consumers/shopping/returns/index_en.htm : read. 14-day doorstep withdrawal; urgent repairs excluded, plumber example.
- https://www.legislation.gov.uk/eudr/2011/7/article/3/data.htm : snippet. Late Payment Directive: 60 days, reference rate plus at least 8 points.
- https://single-market-economy.ec.europa.eu/smes/sme-strategy/late-payment-directive_en : snippet. 40 euro recovery cost.
- https://www.gassaferegister.co.uk/gas-safety/home-gas-safety/home-improvements/ : snippet. Illegal to do gas work unless registered; both parties at fault on sign-off of unregistered work (detail cut for length).
- https://www.hse.gov.uk/gas/domestic/faqs.htm : snippet. Domestic gas work by a registered engineer; checking the register.

### Written from memory, not checked live in this run
WebSearch hit its per-turn limit partway through, so these rest on knowledge I am certain of rather than a page opened today. A reviewer should spot-check them first.
- Curriculum: Wiggins and McTighe stages; Anderson and Krathwohl; Biggs; Black and Wiliam 1998; Dunlosky et al. 2013; Pashler et al. 2008; Tomlinson; UDL; CEFR; EQF; AI Act Annex III point 3.
- Literature: GRADE domains, PRISMA-S, PRISMA-ScR, PRISMA-P, SWiM, QUADAS-2, the funnel plot rule of about 10 studies.
- Statistics: about 14% false positives at five looks (Armitage et al. 1969), Holm and Benjamini-Hochberg, the 16 x variance / effect squared rule, the design effect formula, Twyman's law.
- Grants: Expertise and Resources judged for sufficiency; Form 990.
- Newsroom: Defamation Act 2013 sections, New York Times v. Sullivan, statutory right of reply in France and Germany, criminal defamation in EU states, First Draft's five pillars, AI detector weakness. Searches for the Bellingcat and AFP guides returned nothing useful.
- Translation: ISO 17100 translator evidence (degree; degree plus 2 years; 5 years) is from memory because the fetched pages did not state it; ISO 18587 definitions of full and light post-editing; CLDR plural counts; which countries use sworn translators.
- Video: Pelham C-476/17, InfoSoc Article 5(3), fair use and fair dealing, Content ID behaviour, YouTube chapter rules and synthetic media disclosure, Apple's -16 LKFS. The instructions tell the agent to confirm platform specs live.
- Construction: AACE 56R-08 as the building addendum, NEC4, VOB/B and VOB/C, NRM2, CESMM4.
- Hospitality: the 70% popularity rule (no page found stated it); Working Time Directive 2003/88 figures; Article 9(1)(c) as the allergen item; UK ingredient labelling for food prepacked for direct sale; RevPAR and GOPPAR.
- Field service: Directive 2011/83 Articles 9, 10, 14(4), 16(h); UK Consumer Contracts Regulations 2013; UK Late Payment Act rate; VAT Directive Article 226; Part P; the 0800 111 999 gas emergency line.

---

## Batch: growth


### Email sender rules (lifecycle-email, sdr-prospector)
- https://support.google.com/a/answer/81126 : Gmail requirements for all and bulk senders, 0.3% and 0.1% spam rate, DMARC p=none, one-click, 4.7.28 handling.
- https://support.google.com/a/answer/14229414 : Gmail FAQ: near 5,000 in 24 hours, permanent bulk status, primary domain counting, 48 hour unsubscribe, 7 days below 0.3% for mitigation, transactional exempt.
- https://senders.yahooinc.com/best-practices/ : Yahoo: SPF, DKIM, DMARC, under 0.3% on inbox-delivered mail, 2 day unsubscribe, no published bulk volume.
- https://substrate.office.com/ip-domain-management-snds/postmaster/Policies : Microsoft: over 5,000 a day, SPF, DKIM, DMARC at least p=none, 5 May 2025, junk then 550 5.7.515, no complaint threshold.
- https://www.twilio.com/en-us/blog/insights/gmail-postmaster-tools-changes : Postmaster Tools v1 retirement, reputation views removed, spam data only when mail reaches the inbox (secondary, quotes Gmail).

### B2B and marketing email law (lifecycle-email, sdr-prospector, pr-manager)
- https://www.gesetze-im-internet.de/uwg_2004/__7.html : Germany section 7 UWG: prior express consent, 7(3) customer exception, phone presumed consent.
- https://www.ihk-muenchen.de/email-marketing/ : IHK Munich: no B2B and B2C distinction for email advertising (search result excerpt).
- https://www.ris.bka.gv.at/eli/bgbl/i/2021/190/P174/NOR40238632 : Austria section 174 TKG 2021 (search result).
- https://bmf.gv.at/en/topics/Telecom-and-Postal-Services/Telecommunications-Law-and-Policy/Unsolicited-messages.html : Austria: consent also for the corporate sector, customer exception (search result excerpt).
- https://www.wko.at/service/wirtschaftsrecht-gewerberecht/E-Mails_versenden_-_aber_richtig.html : Austria: fines up to EUR 50,000 (search result excerpt).
- https://www.cnil.fr/fr/la-prospection-commerciale-par-courrier-electronique : CNIL: professionals, subject tied to job, IT director example, opt-out, generic addresses.
- https://www.leto.legal/news/cnil-communications-electroniques-prospection-regles-2026 : CNIL sheet updated 2026, teams overestimate B2B room (secondary).
- https://www.leptidigital.fr/growth-hacking/appels-a-froid-e-mails-pro-devines-cnil-encadre-deux-pratiques-courantes-prospection-commerciale-92263/ : headline only that CNIL addressed guessed professional addresses; page fetch was refused, details not read.
- https://ico.org.uk/for-organisations/direct-marketing-and-privacy-and-electronic-communications/guidance-on-direct-marketing-using-electronic-mail/how-do-we-comply-with-the-pecr-electronic-mail-marketing-rules/ : ICO: corporate subscribers, sole traders as individuals, soft opt-in, charity soft opt-in (search result excerpts; direct fetch refused).
- https://www.ftc.gov/business-guidance/resources/can-spam-act-compliance-guide-business : CAN-SPAM: no B2B exception, 10 business days, postal address.
- https://www.matheson.com/insights/cjeu-clarifies-direct-marketing-rules-under-eprivacy-directive/ : CJEU C-654/23 Inteligo Media, 13 November 2025, soft opt-in on free newsletter accounts (secondary).
- https://cm.twobirds.com/en/insights/2025/understanding-soft-opt-in-when-free-deals-count-as-consent-under-eprivacy-rules : same case, specific to the tiered model (secondary).

### Search and AI answers (seo-strategist)
- https://web.dev/articles/vitals : Core Web Vitals, 75th percentile, all three must pass.
- https://web.dev/articles/lcp : LCP bands.
- https://web.dev/articles/inp : INP bands.
- https://web.dev/articles/cls : CLS bands.
- https://developers.google.com/search/docs/appearance/page-experience : good vitals do not guarantee rankings.
- https://developers.google.com/search/docs/appearance/ai-features : no additional requirements, no special schema or AI text files, Web search type, preview controls, Google-Extended.
- https://developers.google.com/search/blog/2025/05/succeeding-in-ai-search : higher quality clicks claim, measuring value.
- https://developers.google.com/search/docs/appearance/structured-data/search-gallery : current gallery, FAQ and HowTo not listed.
- https://developers.openai.com/api/docs/bots : OAI-SearchBot against GPTBot.
- https://arxiv.org/abs/2311.09735 : GEO paper, KDD 2024, up to 40% on a benchmark.
- https://www.pewresearch.org/short-reads/2025/07/22/google-users-are-less-likely-to-click-on-links-when-an-ai-summary-appears-in-the-results/ : 8% against 15% clicks, March 2025 US panel.
- https://searchengineland.com/bing-webmaster-tools-testing-new-ai-performance-report-468039 : Bing AI Performance report (search result, secondary).
- https://experienceleague.adobe.com/en/docs/brand-visibility-learn/tutorials/measurement/bing-webmaster-tools-ai-performance-report : Bing report scope, not ChatGPT (search result excerpt).

### Consent and analytics (paid-media-analyst)
- https://developers.google.com/tag-platform/security/guides/consent : four parameters, defaults before measurement, choice not stored.
- https://support.google.com/google-ads/answer/10000067 : basic against advanced, general against advertiser-specific model.
- https://support.google.com/analytics/answer/9976101 : cookieless pings, no modelled data without advanced.
- https://support.google.com/analytics/answer/11161109 : behavioural modelling thresholds.
- https://support.google.com/analytics/answer/10596866 : GA4 attribution models, removal in November 2023.
- https://support.google.com/analytics/answer/7667196 : GA4 retention, explorations only.
- https://www.google.com/about/company/user-consent-policy/ : EU user consent policy scope and duties.
- https://support.google.com/google-ads/answer/14310715 : Customer Match EEA consent signals.

### Marketing claims (conversion-copywriter)
- https://legislation.gov.uk/eudr/2005/29/article/12/data.html : UCPD Article 12 (search result excerpt).
- https://legislation.gov.uk/eudr/2006/114/article/7 : Directive 2006/114 Article 7.
- https://www.legislation.gov.uk/eudr/2019/2161/article/3 : UCPD Article 7(6), Annex I 23b and 23c.
- https://www.lw.com/en/insights/european-commission-announces-intention-to-withdraw-eu-green-claims-directive-proposal : Green Claims Directive withdrawal announcement (secondary).
- https://www.algoodbody.com/insights-publications/goodbye-green-claims-directive-where-does-that-leave-the-law-on-environmental-claims : Directive 2024/825 dates and generic claims (secondary, search result excerpt).

### Tenders (rfp-responder)
- https://www.arthurcox.com/insights/public-procurement-thresholds-for-2026-2027/ : 2026 and 2027 thresholds under Directive 2014/24 (secondary; EUR-Lex fetch returned no text).
- https://eur-lex.europa.eu/eli/reg_del/2025/2152/oj/eng : Delegated Regulation 2025/2152, cited as the legal basis.
- https://ted.europa.eu/simap/eforms : eForms, mandatory October 2023, change notices.
- https://single-market-economy.ec.europa.eu/single-market/public-procurement/digital-procurement/european-single-procurement-document-and-ecertis_en : ESPD, only winner provides documents, service closed 2 May 2019, eCertis.
- https://legislation.gov.uk/eudr/2014/24/article/27 : open procedure time limits.
- https://legislation.gov.uk/eudr/2014/24/article/47 : six day rule.
- https://legislation.gov.uk/eudr/2014/24/article/55 : debrief within 15 days.
- https://legislation.gov.uk/eudr/2014/24/article/58 : turnover cap.
- https://www.legislation.gov.uk/eudr/1989/665/article/2a : standstill of 10 calendar days.

### Sales (sales-call-coach, sdr-prospector)
- https://meddicc.com/ : MEDDPICC elements as defined by the site.
- https://meddicc.com/meddpicc-sales-methodology-and-process : Economic Buyer, Champion, Paper Process, Implicate the Pain.
- https://www.huthwaiteinternational.com/spin-selling : SPIN research base of 35,000 calls.
- https://www.gesetze-im-internet.de/stgb/__201.html : Germany section 201 StGB on recording.
- https://artificialintelligenceact.eu/article/5/ : AI Act Article 5(1)(f), emotion inference at work.

### Customer success and support (customer-success, support-specialist)
- https://digital-strategy.ec.europa.eu/en/factpages/data-act-explained : Data Act dates, switching charges removed from 12 January 2027, export format.
- https://www.mccarthy.ca/en/insights/blogs/techlex/moffatt-v-air-canada-misrepresentation-ai-chatbot : Moffatt reasoning and separate entity argument (decision text itself refused the fetch).
- https://fosterandcompany.com/?p=5709 : Moffatt: tribunal on cross-checking the website.
- https://decisions.civilresolutionbc.ca/crt/crtd/en/525448/1/document.do : the decision; fetch refused, cited for the reader.
- https://www.mccannfitzgerald.com/knowledge/technology-and-innovation/one-month-to-go-eu-ai-act-transparency-compliance : AI Act Article 50(1) from 2 August 2026, not moved by the omnibus (secondary).
- https://cms.law/en/swe/legal-updates/new-withdrawal-button-required-on-swedish-e-commerce-platforms-from-19-june-2026 : withdrawal function from 19 June 2026 (search result, secondary).

### Agency (account-manager)
- https://www.legislation.gov.uk/eudr/2011/7/article/3 : Late Payment Directive 60 day rule.
- https://www.gesetze-im-internet.de/urhg/__29.html : section 29 UrhG, copyright not transferable.

### Written from certain knowledge, not re-checked this session
GDPR Articles 6(1)(f), 12(3), 14, 21, 28, 33, 34 and Recital 47. Directive 2014/24 Articles 57(4)(h), 63, 67, 69. Directive 2006/114 Article 4. UCPD Annex I point 7. Price Indication Directive Article 6a. Directive 2019/771 two-year guarantee. Data Act Article 25 two-month notice and 30-day transition. UK Procurement Act 2023 commencement. DMCC Act 2024 fake review ban and CMA fining. FTC reviews rule. MAR Article 17. Section 87 BetrVG. Barcelona Principles on AVEs. Rackham's four call outcomes and SPIN question types. Salesforce forecast categories. CNIL's sanction of Kaspr. Moffatt damages figure of C$650.88 appears only in secondary reports.

---

## Batch: legal-consulting


Status key: OPENED = page or file read directly. SNIPPET = seen in search results only, page not opened.

### GDPR and privacy
- https://gdpr-info.eu/art-33-gdpr/ OPENED: Art. 33(1), (2), (4), (5) wording. Mirror of the text; EUR-Lex returned empty pages to the fetcher and 0 bytes to curl, so the official page itself could not be read.
- https://gdpr-info.eu/art-12-gdpr/ OPENED: Art. 12(3) one month plus two, 12(5) burden on controller, 12(6) identity. Same mirror caveat.
- https://gdpr-info.eu/art-30-gdpr/ OPENED: Art. 30(5) under-250 exemption and its exceptions. Same mirror caveat.
- https://www.edpb.europa.eu/our-work-tools/our-documents/guidelines/guidelines-92022-personal-data-breach-notification-under_en OPENED: Guidelines 9/2022 exist, final version 2.0.
- https://legal247.no/en/insights/digital-omnibus-gdpr/ OPENED: Council mandate ST 14756/26 of 21 Oct 2026, high-risk threshold, 96 hours, Parliament position pending, 18-month application. Law firm summary.
- https://www.consilium.europa.eu/en/press/press-releases/2026/10/21/digital-rulebook-council-agrees-position-to-simplify-data-protection-privacy-and-cyber-reporting-rules/ SNIPPET (403 on fetch): Council position exists, negotiations wait for Parliament.
- https://www.edpb.europa.eu/system/files/documents/2026-02/edpb_edps_jointopinion_202602_digitalomnibus_en.pdf SNIPPET: EDPB and EDPS view on breach changes and personal data definition.
- https://www.wilmerhale.com/en/insights/blogs/wilmerhale-privacy-and-cybersecurity-law/20251201-european-court-of-justice-to-review-challenge-to-eu-us-data-privacy-framework SNIPPET: Latombe appeal to the CJEU.
- https://itdaily.com/news/business/microsoft-latombe-eu-us-data-privacy-framework/ SNIPPET: C-703/25 P pending as of mid-2026.
- https://www.freeths.co.uk/insights-events/legal-articles/2026/european-commission-renews-uk-adequacy-decisions-until-27-december-2031/ SNIPPET: UK adequacy to 27 December 2031.
- https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/subject-access-requests/a-guide-to-subject-access/ SNIPPET: reasonable and proportionate search.
- https://www.ashfords.co.uk/insights/articles/the-data-use-and-access-act-changes-for-data-subject-access-requests SNIPPET: complaints duty from 19 June 2026.
- https://thelens.slaughterandmay.com/post/102m1ip/ico-publishes-new-guidance-on-data-subject-access-requests SNIPPET: stop-the-clock in ICO guidance.

### Fabricated citations
- https://caselaw.nationalarchives.gov.uk/ewhc/admin/2025/1383 OPENED: Ayinde and Al-Haroun, quoted wording (para 6), duty to check (para 7), leadership (para 9), contempt threshold (paras 68 to 69), 18 of 45 authorities (para 74).
- https://www.damiencharlotin.com/hallucinations/ OPENED: 2,262 decisions as of 2 November 2026, with UK and EU member state counts.
- https://www.vaquill.ai/blog/ai-hallucination-sanctions-tracker OPENED: Mata $5,000, Couvrette v. Wisnovsky $110,204.38 total, Johnson v. Dunn disqualification. Vendor tracker; the orders themselves were not opened.

### Contracts
- https://www.mishcon.com/news/eu-data-act-what-cloud-service-vendors-need-to-know-about-new-switching-rights SNIPPET: Data Act switching rules.
- https://www.lewissilkin.com/insights/2025/09/22/the-eu-data-act-whats-it-for-and-what-do-i-have-to-do-102l68x SNIPPET: Data Act application from 12 September 2025, Art. 13.
- https://www.dlapiper.com/en-ae/insights/publications/2025/07/understanding-switching-termination-rights-under-the-data-act SNIPPET: switching and charges from 12 January 2027.
- https://www.harvey.ai/blog/contract-redlining-best-practices SNIPPET: compare to prior version, untracked edits as the main risk. Vendor.
- https://www.sirion.ai/de/library/contract-negotiation/how-to-redline-in-word/ SNIPPET: Document Inspector for metadata. Vendor.
- https://ironcladapp.com/resources/articles/modern-contract-playbook SNIPPET: preferred, fallback, walk-away structure. Vendor.
- https://www.hyperstart.com/blog/contract-playbook/ SNIPPET: gathering positions from templates and past redlines. Vendor.

### Trademarks
- https://www.uspto.gov/trademarks/search OPENED: Trademark Search system at tmsearch.uspto.gov.
- https://intellectual-property-helpdesk.ec.europa.eu/news-events/news/new-edition-nice-classification-enters-force-2026-2026-01-13_en OPENED: Nice 13th edition from 1 January 2026, class moves, TMclass and Harmonised Database.
- https://cms.law/en/int/legal-updates/life-jackets-for-pets-key-changes-in-the-nice-classification-13th-edition SNIPPET: existing registrations not automatically reclassified.
- https://www.euipo.europa.eu/en/trade-marks/before-applying/availability NOT READ (403): EUIPO tool names come from general knowledge and the roster.

### Corporate secretary
- https://www.gov.uk/guidance/verifying-your-identity-for-companies-house OPENED: personal code in next confirmation statement, filing blocked, PSC 14-day windows, offences.
- https://www.pinsentmasons.com/out-law/news/uk-directors-spotlight-id-verification-deadline-looms SNIPPET: transition end in November 2026. Sources differ between 17 and 18 November, so the text says November and tells the agent to confirm.
- https://www.gov.uk/government/publications/late-filing-penalties/late-filing-penalties OPENED: penalties, 9 and 6 months, doubling rule (page updated 16 January 2026).
- https://www.legislation.gov.uk/ukpga/2006/46/section/355 OPENED: records kept at least ten years.
- https://www.legislation.gov.uk/ukpga/2006/46/section/853A OPENED: confirmation statement, 14 days after review period.
- https://www.gesetze-im-internet.de/gmbhg/__42a.html OPENED: 8 and 11 months for adopting accounts.
- https://www.gesetze-im-internet.de/hgb/__325.html OPENED: disclosure within 12 months.
- https://corp.delaware.gov/paytaxes/ OPENED: 1 March due date, $200 penalty, 1.5% monthly interest.
- https://www.fincen.gov/boi OPENED: final rule of 11 August 2026 exempts US companies; foreign-entity deadlines described under the March 2025 rule.
- https://www.mofo.com/resources/insights/260820-fincen-ends-beneficial-ownership-reporting SNIPPET: August 2026 final rule.
- https://www.transparency.org/en/news/countdown-to-new-eu-beneficial-ownership-rules SNIPPET: AMLD6 legitimate-interest access, deadline 10 July 2026, uneven progress.
- https://eucrim.eu/news/new-anti-money-laundering-directive-amld-6/ SNIPPET: AMLD6 content.

### Intake and litigation
- https://www.sra.org.uk/solicitors/guidance/conflicts-interest/ OPENED: paras 6.1 and 6.2, exceptions, written informed consent, prospective clients count.
- https://www.gesetze-im-internet.de/kschg/__4.html OPENED: three weeks from receipt of written dismissal.
- https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part-57a-business-and-property-courts/practice-direction-57ad-disclosure-in-the-business-and-property-courts OPENED: preservation duties, known adverse documents, whether or not an order is made.

### Consulting
- ICAEW Corporate Finance Faculty, Best-Practice Guideline 72, Commercial Due Diligence (2025): https://www.icaew.com/-/media/corporate/files/technical/corporate-finance/guidelines/icaew-commercial-due-diligence-guideline.ashx OPENED, pages 1 to 21: forms of CDD, no assurance, referencing limits, clean teams, auction limits on customer contact, MAR, hypothesis-driven approach.
- https://slideworks.io/resources/mckinsey-problem-solving-process SNIPPET: seven-step process.
- https://www.mckinsey.com/capabilities/strategy-and-corporate-finance/our-insights/how-to-master-the-seven-step-problem-solving-process NOT READ (timed out twice).
- https://thinkinsights.net/playbooks/problem-structuring-hypothesis-driven-consulting-playbook-issue-trees-mece SNIPPET: issue tree against hypothesis tree.
- https://deckary.com/blog/pyramid-principle-consulting SNIPPET: titles test.
- https://thinkinsights.net/insights/minto-pyramid-principle SNIPPET: answer-first, Minto origin.

### Written from established knowledge, not opened in this session
The web search budget for the turn ran out before these could be checked, and several official pages refused the fetcher. Each is stable or carries a "check" instruction in the text.
- GDPR Arts. 15, 28(3), 28(10), 30(1) and (2), 34, 35, 36, 49; WP248 rev.01 criteria; CJEU C-307/22, C-154/21, C-487/21, C-252/21, C-311/18; T-553/23; EDPB Recommendations 01/2020; SCC Decisions 2021/914 and 2021/915; CCPA 45-day period.
- EUTMR Arts. 7, 8, 18, 46, 58 and priority and grace periods; TMA 1994 ss.5 and 46; TM7A; US 30-day opposition; DuPont and Abercrombie; SABEL, Canon, Lloyd, Lindt, SkyKick.
- Regulation (EU) 2024/1624 application date (one search snippet agrees) and its "25% or more" wording; Directive 2013/34 Art. 30; § 15 and § 40 GmbHG; § 20 GwG; UK PSC conditions; no AGM for UK private companies.
- UCTA 1977; §§ 305 to 310 BGB; Data Act two-month and 30-day details are deliberately left out of the final text.
- MLR 2017 reg. 28; LSAG guidance; ABA Model Rules 1.7 to 1.10 and 1.18 (ABA page returned 403); CCBE Code 3.2; Employment Rights Act 2025 extending UK tribunal limits (timing unverified, so the text tells the agent to check Acas).
- Three Rivers No 5; SFO v ENRC; Akzo Nobel C-550/07 P; English court guides on neutral chronologies (Chancery Guide page returned 404).
- Minto, Zelazny, Rumelt, Martin, Admiralty source grading, NRR and GRR definitions, 3:1 LTV to CAC rule of thumb.

---

## Batch: people-tech


### Pay transparency (recruiter, comp-analyst)
- https://iuslaboris.com/insights/eu-pay-transparency-directive-which-countries-have-transposed/ : tracker dated 30 Sept 2026; Italy and Slovakia transposed, France, Spain, Netherlands in draft, Germany and Ireland initial steps; 7 June 2026 deadline.
- https://corp-intl.com/news/which-countries-implemented-eu-pay-transparency-directive : Sept 2026 count of five fully transposed (Slovakia, Lithuania, Italy, Malta, Greece); disagrees with other trackers on Malta.
- https://www.forvismazars.com/ro/en/insights/blog/business-insights/pay-transparency-obligations-risks : Art. 5, 6, 7, 9, 10, limitation period, burden of proof, compensation, reporting tiers and dates, as summarised by a firm (EUR-Lex fetch returned empty).
- https://ogletree.com/insights-resources/blog-posts/eu-pay-transparency-directive-updates-on-implementation-across-member-states/ : member states may exceed the minimum; national examples (dated May 2025).

### AI Act and Digital Omnibus (recruiter, learning-designer)
- https://artificialintelligenceact.eu/annex/3/ : Annex III point 4 wording on job ads, filtering applications, evaluating candidates.
- https://artificialintelligenceact.eu/article/26/ : deployer duties, human oversight, logs six months, Art. 26(7) and 26(11) notices.
- https://artificialintelligenceact.eu/article/5/ : Art. 5(1)(f) emotion inference at the workplace.
- https://www.nicfab.eu/en/posts/digital-omnibus-ai-official-journal/ : Regulation (EU) 2026/1744, OJ 24 July 2026, in force 27 July 2026, Annex III from 2 December 2027, Art. 4 replaced ("support" AI literacy).
- https://www.actuia.com/en/news/ai-act-obligations-in-force-and-delays-under-omnibus-2026-1744/ : same dates and Art. 4 wording.
- https://www.mayerbrown.com/en/insights/publications/2026/07/eu-ai-act-news-digital-omnibus-on-ai-new-guidance-on-risk-classification-gpai-and-transparency-obligations : draft Commission high-risk guidelines of 19 May 2026; ranking tools with primary-input scores are high-risk.

### Recruiting evidence and retention (recruiter)
- https://www.siop.org/tip-article/is-cognitive-ability-the-best-predictor-of-job-performance : Sackett et al. (2022) estimates: structured interviews .42, job knowledge .40, work sample .33, cognitive ability .31.
- https://www.lexgo.be/en/news-and-articles/14859-how-long-should-hr-data-be-retained-when-the-law-is-silent-the-cnil-guidelines-as-a-useful-reference-framework : CNIL up to two years from last contact for a CV bank.
- https://join.com/en/blog/gdpr-compliance-in-recruiting : GDPR sets no fixed period; 6 to 12 months common in practice (vendor source, used only as "many firms").

### HR law (hr-advisor)
- https://www.legislation.gov.uk/eudr/2019/1937/article/8 and /article/9 : whistleblowing 50-worker threshold, seven days, three months.
- https://www.legislation.gov.uk/eudr/2019/1152/article/5 and /article/8 : written terms by seventh calendar day and one month; probation six months.
- https://www.legislation.gov.uk/eudr/2019/1158/article/4 , /article/5 , /article/6 : paternity 10 working days, parental four months with two non-transferable, carers five days.
- https://www.acas.org.uk/acas-code-of-practice-on-disciplinary-and-grievance-procedures/html : disciplinary steps, right to be accompanied, tribunal adjustment up to 25%.
- https://knowledge.dlapiper.com/dlapiperknowledge/globalemploymentlatestdevelopments/2026/revised-timeline-for-implementation-of-the-employment-rights-act-2025 : Employment Rights Act 2025 timetable (October 2026, 1 January 2027).

### Learning design (learning-designer)
- https://www.kirkpatrickpartners.com/the-kirkpatrick-model/ : four-level definitions, Return on Expectations, performance environment.
- https://wikicap.ulb.be/images/5/5f/Molenda_2003.pdf and https://en.wikipedia.org/wiki/ADDIE_model : Molenda on ADDIE as an umbrella label, Florida State 1975.
- https://learn.alleninteractions.com/services/custom-learning/sam/elearning-development : SAM phases, Savvy Start, Design Proof, Alpha, Beta, Gold.

### Product (product-manager)
- https://www.intercom.com/blog/rice-simple-prioritization-for-product-managers/ : RICE definitions, scales, not a hard rule, dependencies and table stakes.
- https://www.producttalk.org/opportunity-solution-trees/ : opportunity solution tree levels, definition of opportunity, assumption tests.
- https://www.svpg.com/four-big-risks/ : Cagan's four risks.

### UX research (ux-researcher)
- https://www.thematicanalysis.net/doing-reflexive-ta/ : six phases, definition of a theme.
- https://www.nngroup.com/articles/why-you-only-need-to-test-with-5-users/ : 85%, L = 31%, three rounds of five, per-group numbers, 20 for quantitative.
- https://www.nngroup.com/articles/how-to-rate-the-severity-of-usability-problems/ : severity 0 to 4, frequency, impact, persistence, three raters.

### Accessibility (accessibility-auditor)
- https://www.w3.org/WAI/standards-guidelines/wcag/new-in-22/ : nine new criteria, levels, 4.1.1 removed, 24 by 24 CSS px, Recommendation 5 October 2023.
- https://www.w3.org/WAI/test-evaluate/conformance/wcag-em/ : WCAG-EM 2 published 23 July 2026, steps, scope extended to apps.
- https://accessible-eu-centre.ec.europa.eu/content-corner/news/european-accessibility-standard-en-301-549-has-been-updated-2026-09-07_en : EN 301 549 v4.1.1 adopts WCAG 2.2; v3.2.1 stays the legal reference until cited in the OJ.
- https://commission.europa.eu/strategy-and-policy/policies/justice-and-fundamental-rights/disability/european-accessibility-act-eaa_en : EAA products and services in scope.
- https://www.legislation.gov.uk/eudr/2019/882/article/3 and /article/4 : microenterprise definition, Art. 4(5) service exemption.
- https://oeil.europarl.europa.eu/oeil/en/document-summary?id=1584985 : Art. 32 transition to 28 June 2030.
- https://www.deque.com/blog/automated-testing-study-identifies-57-percent-of-digital-accessibility-issues : 57% by volume, about 2,000 audits, 16 of 50 AA criteria.
- https://accessibility.blog.gov.uk/2017/02/24/what-we-found-when-we-tested-tools-on-the-worlds-least-accessible-webpage/ : 143 barriers, best tool 41%, combined 71%, 29% missed by all.
- https://www.ada.gov/resources/2024-03-08-web-rule/ : ADA Title II WCAG 2.1 AA, dates extended to April 2027 and April 2028.
- https://ftc.gov/news-events/news/press-releases/2025/04/ftc-approves-final-order-requiring-accessibe-pay-1-million : FTC final order, $1 million.

### Analytics engineering (analytics-engineer)
- https://docs.getdbt.com/best-practices/how-we-structure/1-guide-overview : source-conformed to business-conformed, layers.
- https://docs.getdbt.com/best-practices/how-we-structure/2-staging : staging naming, one-to-one, no joins or aggregation, views.
- https://docs.getdbt.com/best-practices/how-we-structure/4-marts : grain, naming, materialisation order, 4 or 5 concepts, Semantic Layer reversal.
- https://docs.getdbt.com/docs/build/data-tests : four generic tests, data_tests key, arguments from v1.10.5, singular tests.

### SRE (sre-incident)
- https://sre.google/sre-book/managing-incidents/ : roles, when to declare, live document, handoff wording, priorities.
- https://sre.google/sre-book/postmortem-culture/ : postmortem triggers, blameless definition.
- https://sre.google/workbook/error-budget-policy/ : example policy, four-week window, 20% threshold, P0 action.
- https://sre.google/workbook/alerting-on-slos/ : error budget and burn-rate formulas, multi-window alert table.
- https://response.pagerduty.com/before/severity_levels/ : SEV-1 to SEV-5, take the higher when unsure.
- https://www.springlex.eu/en/packages/nis2/nis2-directive/article-23/ : NIS2 Art. 23 deadlines and significant incident.

### Technical writing (technical-writer)
- https://diataxis.fr/ and https://diataxis.fr/map/ : four forms, axes, placement of each.
- https://keepachangelog.com/en/1.1.0/ : changelog principles and change types.

### Policy (policy-analyst)
- https://en.wikipedia.org/wiki/Eightfold_path_(policy_analysis) : the eight steps, Patashnik co-authorship (book itself not read).
- https://commission.europa.eu/law/law-making-process/planning-and-proposing-law/better-regulation/better-regulation-guidelines-and-toolbox_en : Guidelines 3 November 2021, Toolbox 1 December 2025.
- https://commission.europa.eu/law/law-making-process/planning-and-proposing-law/better-regulation/better-regulation-guidelines-and-toolbox/better-regulation-toolbox_en : chapters and tool numbers (#5, #7, #21, #23, #47, #59).
- https://www.bruegel.org/first-glance/european-commissions-recommitment-better-regulation-yet-be-proven : 28 April 2026 Communication, major vs targeted, Scrutiny Board, urgency derogation (commentary, Communication itself not read).

### Donors (donor-relations)
- https://www.legislation.gov.uk/eudr/2002/58/article/13 : ePrivacy Art. 13 consent and similar-products exception; no mention of charities.
- https://gdpr-info.eu/art-9-gdpr/ : Art. 9(2)(d) conditions.
- https://www.edpb.europa.eu/system/files/2024-10/edpb_guidelines_202401_legitimateinterest_en.pdf : Recital 47 does not make direct marketing automatically a legitimate interest (consultation version; final status not confirmed).
- https://ico.org.uk/for-organisations/direct-marketing-and-privacy-and-electronic-communications/guidance-on-direct-marketing-using-electronic-mail/ : charitable purposes soft opt-in, reg. 22(3A), conditions.
- https://www.lewissilkin.com/insights/2026/05/08/the-charitable-purposes-soft-opt-in-ico-issues-updated-direct-marketing-guidanc-102ms57 : 5 February 2026 start, details collected from that date, UK GDPR basis still needed.
- https://www.civilsociety.co.uk/news/ico-fines-rspca-and-british-heart-foundation-after-daily-mail-expose.html and https://www.rpclegal.com/thinking/data-and-privacy/eleven-more-charities-fined-by-ico-in-wealth-screening-probe/ : ICO fines 2016 and 2017, wealth screening, tele-matching.
- https://www.fundraisingregulator.org.uk/code : 2025 Code in force 1 November 2025, Standard 5.
- https://open-exam-prep.com/study-guides/cfre/relationship-cultivation-moves/cultivation-planning : moves management attribution to Dunlop and Smith at Cornell (study guide, secondary).

### Written from certain knowledge, not fetched this session
GDPR Arts. 4(11), 5(1)(e), 7(3), 13, 15, 17, 21, 22, 25, 28, 33, 35; Directive 2003/88 leave; Directive 2019/1937 Arts. 16, 19; Directive 2023/970 Arts. 3, 4(4), 12, 16, 18, 21 wording; TFEU Art. 157; Rome I Art. 8; CJEU C-55/18 CCOO, C-684/16 Max-Planck; BetrVG s.102; StGB s.201; Equality Act 2010 s.60; NYC Local Law 144; UK gender pay gap 250+ threshold; Mager's objectives; Anderson and Krathwohl 2001; Dunlosky et al. 2013; Pashler et al. 2008; LYBUNT and SYBUNT; EU evaluation criteria; Bardach's criteria list; NN/g's 40-participant quantitative figure; Maryland Scientific Methods Scale.

---

## Batch: personal-a


### EU consumer rights and disputes (life-admin, purchase-advisor, home-fixit)
- https://europa.eu/youreurope/citizens/consumers/shopping/guarantees-returns/index_en.htm : 2-year legal guarantee from receipt, 1-year reversed burden (2 in some countries), second-hand down to 1 year, private sellers excluded (page last checked 05/10/2026).
- https://europa.eu/youreurope/citizens/consumers/shopping/returns/index_en.htm : 14-day withdrawal, start dates, exceptions, return postage, urgent repairs exception (page last checked 28/04/2026; no note on the withdrawal function).
- https://www.gtlaw.com/en/insights/2026/5/eu-consumer-law-new-withdrawal-button-requirements-for-online-contracts : withdrawal function wording, two steps, acknowledgement with date and time, applies from 19 June 2026 (law firm note on the German transposition, secondary).
- https://vbb.com/wp-content/uploads/2026/06/8-06-26-The-online-withdrawal-button.pdf : Art. 11a; Belgium had not transposed by June 2026, so rollout varies (search snippet, secondary).
- https://commission.europa.eu/law/law-topic/consumer-protection-law/directive-repair-goods_en : Right to Repair Directive 2024/1799, applies from 31 July 2026, extra year of legal guarantee when repair is chosen.
- https://www.lewissilkin.com/insights/2026/03/18/what-does-the-right-to-repair-directive-mean-for-your-business : guarantee extension applies to contracts from 31 July 2026 (practitioner note, secondary).
- https://www.eccnet.eu/ : ECC-Net free advice, 29 offices, EU, Norway and Iceland.
- https://europa.eu/youreurope/citizens/consumers/consumers-dispute-resolution/index_en.htm : dispute ladder: trader, ADR, ECC-Net, courts, European Small Claims (page last checked 28/04/2026).
- https://www.eccnederland.nl/en/news/odr-platform-shut-down-19-july-2025 : EU ODR platform shut down in July 2025.
- https://cms.law/en/aut/legal-updates/the-odr-platform-is-closing-what-businesses-need-to-know : ODR closure on 20 July 2025 under Regulation (EU) 2024/3228 (secondary).

### Flights (life-admin, trip-planner)
- https://europa.eu/youreurope/citizens/travel/passenger-rights/air/index_en.htm : current EU261 rules: 3 hours, €250/400/600, extraordinary circumstances, airline then national body after 2 months, national time limits (page last checked 21/09/2026; no note on reform).
- https://oeil.europarl.europa.eu/oeil/en/procedure-file?reference=2013/0072(COD) : reform is Regulation (EU) 2026/2202: Parliament 7 July 2026, Council 13 July 2026, signed 16 September 2026, OJ 2 October 2026.
- https://data.consilium.europa.eu/doc/document/PE-39-2026-REV-2/en/pdf : joint text: delay at arrival exceeding three hours, €250/400/600, nine months to request compensation, 30 calendar days to pay or justify, enters into force on the 20th day after publication, applies 12 months later. I read this text, not the Official Journal version; EUR-Lex returned an empty page. "Around late October 2027" is my calculation from these dates.

### UK and US consumer rules (life-admin, home-fixit)
- https://www.legislation.gov.uk/ukpga/2015/15/section/22?view=plain : Consumer Rights Act 2015, 30-day short-term right to reject.
- https://www.citizensadvice.org.uk/consumer/get-more-help/if-you-need-more-help-about-a-consumer-issue/ : consumer helpline 0808 223 1133, England, what advisers can and cannot do.
- https://www.citizensadvice.org.uk/consumer/somethings-gone-wrong-with-a-purchase/getting-your-money-back-if-you-paid-by-card-or-paypal/ : Section 75 limits (£100 to £30,000 per item), chargeback, PayPal.
- https://www.ftc.gov/legal-library/browse/rules/cooling-period-sales-made-home-or-other-locations : FTC Cooling-Off Rule, 3 business days, door-to-door and temporary locations.
- https://www.ftc.gov/business-guidance/resources/businesspersons-guide-federal-warranty-law : Magnuson-Moss, implied warranties, no general federal return right described.

### Scams (tech-helper, investing-explainer, tax-organizer)
- https://www.citizensadvice.org.uk/consumer/scams/check-if-something-might-be-a-scam/ : scam signs, 159, payee checks, Have I Been Pwned.
- https://www.citizensadvice.org.uk/consumer/scams/what-to-do-if-youve-been-scammed/ : steps after a scam, third-party mandate (page last reviewed 2019, used only for general steps).
- https://www.reportfraud.police.uk/news/report-fraud-service-goes-live/ : Report Fraud replaced Action Fraud.
- https://www.cityoflondon.police.uk/news/city-of-london/news/2025/december/report-fraud-service-goes-live-with-full-public-launch-in-january-2026/ : launch dates, 0300 123 2040, coverage of England, Wales and Northern Ireland, Scotland uses 101.
- https://consumer.ftc.gov/articles/what-do-if-you-were-scammed : what to do by payment method, ReportFraud.ftc.gov, IdentityTheft.gov.
- https://www.ncsc.gov.uk/collection/top-tips-for-staying-secure-online : three random words, separate email password, 2-step verification, password managers, backups, updates.
- https://www.ncsc.gov.uk/collection/phishing-scams : reporting scam messages (links to the per-channel pages).
- https://finextra.com/newsarticle/44675/psr-to-cut-app-reimbursement-limit-to-85000 : UK transfer scam limit of £85,000 (2024, secondary; written as "when last checked" with an instruction to confirm).
- https://www.hlc.com/en/publications/app-fraud-uk-psr-confirms-introduction-of-world-first-reimbursement-requirement : 13-month claim window (2024, secondary).
- https://techinsights.linklaters.com/post/102lw95/psd3-breakthrough-eu-legislators-agree-payments-regulation-reforms : EU payments deal, payee checks and fraud liability (secondary; I could not confirm publication, so written as "agreed but not yet applying").
- https://www.pwc.com/cz/cs/blog/rizeni-rizik/psd-3-the-psr-payments-reform.html : verification of payee mandatory in the eurozone from 9 October 2025 (secondary).

### Debt and budgeting (household-budget)
- https://www.stepchange.org/contact-us.aspx : StepChange 0800 138 1111, free.
- https://nationaldebtline.org : National Debtline free advice.
- https://www.nfcc.org/ : NFCC nonprofit counselling, 877-360-6322. The page does not say the service is always free, so the text says to ask about fees.
- https://lawplayer.com/eu/article/32023L2225/art_36 : Consumer Credit Directive 2023/2225 Art. 36, independent debt advice with limited charges (text mirror, secondary).

### Investing (investing-explainer)
- https://www.esma.europa.eu/investor-corner : red flags (unexpected contact, pressure, deepfakes, clone sites, admin fees, pyramid schemes), ESMA register, national regulators.
- https://www.fca.org.uk/scamsmart/how-avoid-investment-scams : red flags, FCA Firm Checker, Warning List, helpline 0800 111 6768, Report Fraud.
- https://www.investor.gov/introduction-investing/general-resources/news-alerts/alerts-bulletins/investor-bulletins/updated : SEC fee example, $100,000 at 4 percent for 20 years ($208,000, $198,000, $179,000). Search snippet; the page itself was not fetched.
- https://www.fscs.org.uk/what-we-cover/ : FSCS £85,000 for investments, £120,000 for deposits on failures after 30 November 2025.
- https://www.gov.uk/individual-savings-accounts : ISA allowance £20,000 for 2026 to 2027.

### Life organizing (life-organizer)
- https://gettingthingsdone.com/what-is-gtd/ : Allen's five steps: capture, clarify, organize, reflect, engage. Note: "reflect" rather than the roster's "review".

### Family (family-planner)
- https://inequality.hks.harvard.edu/publications/cognitive-dimension-household-labor : Daminger 2019, American Sociological Review, four parts of cognitive labor and gender findings.

### Home safety (home-fixit)
- https://sgn.co.uk/smellgas : what to do if you smell gas, 0800 111 999.
- https://northerngasnetworks.co.uk/?p=21350 : steps including no electrical switches and care around cellars (search snippet).
- https://www.nhs.uk/conditions/carbon-monoxide-poisoning/ : CO symptoms, when to call 999, 0800 111 999.
- https://www.hse.gov.uk/asbestos/introduction/index.htm : asbestos may be in any building built or refurbished before 2000.
- https://www.hse.gov.uk/asbestos/faq.htm : homeowner DIY position, trained professionals encouraged (search snippet).
- https://www.epa.gov/asbestos/protect-your-family-exposures-asbestos : leave intact material alone, no sanding, drilling or vacuuming, professional sampling.
- https://www.epa.gov/asbestos/protect-your-family-asbestos-contaminated-vermiculite-insulation : assume vermiculite contains asbestos.
- https://www.electricalsafetyfirst.org.uk/guidance/your-questions-answered/questions/what-is-notifiable-work/ : notifiable electrical work.
- https://www.electricalsafetyfirst.org.uk/safety-advice/home-and-people/diy-ers/ : use a registered electrician.
- https://www.tameside.gov.uk/documents/d/guest/note20-pdf : Part P: new circuits, consumer unit, special locations (council guidance, secondary).
- https://consumer.ftc.gov/articles/how-avoid-home-improvement-scam : contractor red flags, written contract, payment stages, 3-day cancellation.

### Buying (purchase-advisor)
- https://www.gov.uk/check-mot-history : MOT history, mileage at each test, tests since 2005.
- https://www.eccnet.eu/publication/mileage-verification-car-pass-ensuring-accuracy-vehicles-abroad : Car-Pass mileage record.
- https://www.fleetnews.co.uk/news/manufacturer-news/2018/06/08/industry-welcomes-eu-move-to-combat-clocking : 5 to 12 percent and 30 to 50 percent clocking estimates (2018 report of an EU estimate, secondary).
- https://cms.law/en/gbr/legal-updates/no-more-faux-five-stars-the-dmcc-act-bans-fake-reviews : UK fake review ban since 6 April 2025 (secondary).

### Travel (trip-planner)
- https://www.gov.uk/foreign-travel-advice : per-country advice and email alerts.
- https://home-affairs.ec.europa.eu/news/entryexit-system-will-become-fully-operational-10-april-2026-2026-03-30_en : EES fully operational on 10 April 2026.
- https://home-affairs.ec.europa.eu/news/main-differences-between-ees-and-etias-what-travellers-need-know-2026-04-28_en : ETIAS planned for the last quarter of 2026, official site only, fake sites. No later announcement found, so the text tells the agent to check whether it is live.
- https://www.gov.uk/eta : UK ETA £20, each traveller including babies, exemptions, imitation sites.

### Tax (tax-organizer)
- https://www.gov.uk/self-assessment-tax-returns/deadlines : deadlines for 2025 to 26.
- https://www.gov.uk/keeping-your-pay-tax-records/how-long-to-keep-your-records : 22 months and 15 months rules.
- https://www.irs.gov/businesses/small-businesses-self-employed/how-long-should-i-keep-records : 3, 6 and 7 year rules, property records.
- https://www.irs.gov/individuals/free-tax-return-preparation-for-qualifying-taxpayers : VITA (about $70,244 or less), TCE (60 and over).
- https://www.kingsleynapley.co.uk/our-insights/articles/crypto-reporting-is-changing-what-this-means-for-you-and-hmrc/ : UK CARF from 1 January 2026 (secondary).
- https://fintelegram.com/dac8-carf-2026-crypto-tax-transparency-switch-is-on/ : EU DAC8 data collection from 1 January 2026 (secondary).

### Written from settled knowledge, not fetched this session
The WebSearch budget for the turn ran out, and some official pages returned 403 or 404. These were not read live. I kept only what I am certain of, and the instructions tell the agent to confirm on the official page.
- Directive 2011/83 Arts. 10, 11, 13: 12-month extension, withdrawal by any clear statement, 14-day refund.
- European Small Claims limit of €5,000 (Regulation 2015/2421), no lawyer needed.
- UK Consumer Rights Act s.19 six-month presumption; Consumer Contracts Regulations 2013 14-day cancellation.
- UK flight rules kept in pounds (CAA page did not give detail, so no amounts are stated).
- Germany's Mahnbescheid two weeks (ZPO s.692); England and Wales county court 14 days.
- UK 7726 and report@phishing.gov.uk; ic3.gov (the FBI's reporting site).
- National Debtline number 0808 808 4000; MABS, German, French and Dutch debt services (no numbers given except UK and US); UK Breathing Space 60 days.
- 50/30/20 (Warren and Tyagi, All Your Worth); Gal and McShane 2012, Journal of Marketing Research; Rodsky, Fair Play.
- GTD details beyond Allen's page: two-minute rule, hard landscape, weekly review's get clear, get current, get creative.
- BRE Digest 251 crack categories; Gas Safe legal requirement; US lead paint 1978; smoke and CO alarm placement.
- Schengen passport validity; EHIC and GHIC scope; Package Travel Directive 2015/2302.
- EU energy label and EPREL; EU rule on the 30-day prior price (Directive 98/6 as amended); EU duty to say whether reviews are checked (Directive 2005/29 as amended).
- SPIVA, PRIIPs KID and its 1 to 7 indicator, US ETFs unavailable to EU retail, EU deposit guarantee €100,000, EU investor compensation €20,000 minimum, US SIPC $500,000, EU CFD loss disclosure, tax wrapper names by country.
- DAC7 platform reporting; UK Making Tax Digital phase-in; HMRC and IRS contact practices; TaxAid, Tax Help for Older People, Lohnsteuerhilfeverein; national tax portal names.

---

## Batch: personal-b


The session's WebSearch budget ran out partway through. The later checks were direct fetches of URLs I knew, on the publishers' own sites. No search engine was queried by other means. Items I could not check are listed at the end.

### AI chatbots, sycophancy, delusions, dependence
- https://www.apa.org/topics/artificial-intelligence-machine-learning/health-advisory-ai-chatbots-wellness-apps-mental-health.pdf : APA health advisory, November 2025, read in full: not a substitute for care, sycophancy, dependence signs, higher-risk groups, unreliable crisis handling, privacy.
- https://www.apa.org/news/press/releases/2025/11/ai-wellness-apps-mental-health : APA press release (page text came back empty; used through search summary only, the PDF above is the source).
- https://www.apa.org/topics/artificial-intelligence-machine-learning/guide-navigating-ai-advice.pdf : APA guide, June 2026: immediate human help for hearing or seeing things or strong distressing beliefs, framing of questions, ask for several options, no major decisions on a chatbot alone.
- https://arxiv.org/abs/2510.01395 and https://doi.org/10.1126/science.aec8352 : Cheng et al., Science 2026: models affirm users nearly 50 percent more than people; one chat lowers willingness to repair and raises conviction.
- https://spirals.stanford.edu/research/characterizing/ : Stanford-led chat log study of 19 people reporting harm: amplification, calling users unique, limits on what it shows.
- https://arxiv.org/abs/2602.19141 : Chandra et al., 2026 simulation of sycophancy and delusional spiralling (background).
- https://arxiv.org/abs/2504.18412 : Moore et al., 2025: models encouraging delusional thinking in simulated therapy (background).
- https://arxiv.org/abs/2503.17473 : Fang et al., four-week randomised study of 981 people: heavier voluntary use and worse outcomes (background).
- https://www.nhs.uk/mental-health/conditions/psychosis/symptoms/ : delusions and hallucinations described, urgent help.
- https://www.nhs.uk/mental-health/conditions/bipolar-disorder/symptoms/ : signs of mania.
- https://www.nhs.uk/nhs-services/mental-health-services/where-to-get-urgent-help-for-mental-health/ : overdose and serious self-harm as emergencies, 111 mental health option, Samaritans, Childline.
- https://www.nhs.uk/mental-health/feelings-symptoms-behaviours/feelings-and-symptoms/low-mood-sadness-depression/ : two-week threshold, self-referral to talking therapies.

### Journaling
- https://Www.Gwern.net/doc/psychology/2006-frattaroli.pdf : Frattaroli 2006, 146 randomised studies, r = .075.
- https://www.cambridge.org/core/journals/advances-in-psychiatric-treatment/article/emotional-and-physical-health-benefits-of-expressive-writing/ED2976A61F5DE56B46F07A1CE9EA9F9F : Baikie and Wilhelm: 15 to 20 minutes, 3 to 5 sessions, short-term distress, caution with abuse survivors and PTSD.

### Medical red flags
- https://www.nhs.uk/conditions/stroke/symptoms/ : FAST and other stroke signs, call even if symptoms stop.
- https://www.nhs.uk/symptoms/chest-pain/ : chest pain signs for 999.
- https://www.nhs.uk/conditions/sepsis/ : sepsis red flags, adults and under 5s.
- https://www.nhs.uk/conditions/meningitis/ : meningitis red flags.
- https://www.nhs.uk/conditions/anaphylaxis/ : signs and first steps, do not stand.
- https://www.nhs.uk/symptoms/confusion/ : sudden confusion as emergency and its causes.
- https://medlineplus.gov/ency/article/001927.htm : wider list of emergency warning signs.
- https://medlineplus.gov/ency/article/000473.htm : rhabdomyolysis symptoms, dark urine.
- https://medlineplus.gov/lab-tests/how-to-understand-your-lab-results/ : reference ranges, false results, questions to ask.

### Habits and sleep
- https://www.behaviormodel.org/ : B=MAP, primary definition.
- https://www.cooksmarts.com/?p=20032 and https://goalsandprogress.com/tiny-habits-fogg-behavior-model-explained/ : Tiny Habits recipe wording (secondary).
- https://kops.uni-konstanz.de/entities/publication/2e749bfb-8533-437c-8203-7e788c910c5f : Gollwitzer and Sheeran 2006, 94 tests, d = .65.
- https://goalsandprogress.com/implementation-intentions-research/ : Sheeran, Listrom and Gollwitzer 2024, 642 tests and moderators (secondary).
- https://www.nhs.uk/conditions/insomnia/ : sleep hours, self-help, when to see a GP, CBT, sleeping pills.
- https://www.nhs.uk/conditions/sleep-apnoea/ : signs, GP, driving.
- https://www.covwarkformulary.nhs.uk/docs/chapter04/CG046-Primary%20Care%20Insomnia%20Management%20Pathway%20for%20Adults.pdf : chronic insomnia threshold, 3 nights, 3 months (NHS pathway citing NICE).
- https://www.oxfordhealthformulary.nhs.uk/docs/bob-icb-advice-for-gps-on-managing-insomnia.pdf : CBT-I first line (NHS pathway citing NICE).

### Fitness
- https://pmc.ncbi.nlm.nih.gov/articles/PMC7719906/ : WHO 2020 guidelines by age group, pregnancy, older adults' balance work.
- https://www.who.int/news-room/fact-sheets/detail/physical-activity : any activity beats none.
- https://eparmedx.com/ : PAR-Q+ current 2025 form, what a yes means, follow-up pages, ePARmed-X+.
- https://form.jotform.com/221436864485060 : a copy of the seven questions and the delay advice (not the official PDF).
- https://pubmed.ncbi.nlm.nih.gov/19204579/ : ACSM progression models: novice loads and frequency, 2 to 10 percent load increase.
- https://www.cdc.gov/physical-activity-basics/measuring/index.html : talk test, 0 to 10 effort scale.

### Food safety and eating
- https://www.gov.uk/government/publications/cooking-your-food/cooking-your-food : FSA 70°C for 2 minutes, what cooks through, steaming hot, reheat once, danger zone.
- https://www.gov.uk/government/publications/how-to-chill-freeze-and-defrost-food-safely/how-to-chill-freeze-and-defrost-food-safely : fridge, freezer, cooling, 48 hours, defrosting.
- https://www.gov.uk/government/publications/home-food-fact-checker/home-food-fact-checker : rice, kidney beans, use-by and best-before.
- https://www.gov.uk/government/publications/food-allergy-and-intolerance-advice-for-consumers/food-allergy-and-intolerance-advice-for-consumers : the 14 allergens, may contain, cross-contact, vegan labels.
- https://www.vdh.virginia.gov/content/uploads/sites/6/2023/07/Cook_to_a_Safe_Minimum_Internal_Temperature.pdf : FoodSafety.gov chart in °F (foodsafety.gov itself returned 403).
- https://www.nhs.uk/pregnancy/keeping-well/foods-to-avoid/ : foods to avoid in pregnancy.
- https://www.nhs.uk/conditions/baby/weaning-and-feeding/foods-to-avoid-giving-babies-and-young-children/ : honey, whole nuts, salt.
- https://www.nhs.uk/mental-health/feelings-symptoms-behaviours/behaviours/eating-disorders/overview/ : warning signs, GP, Beat helpline number.

### Caring for a relative
- https://www.nhs.uk/conditions/dementia/symptoms-and-diagnosis/symptoms/ : early signs, see a GP sooner.
- https://www.who.int/news-room/fact-sheets/detail/dementia : not a normal part of ageing, signs, iSupport for carers.
- https://www.nhs.uk/conditions/falls/ : when to call 999, do not lift, how to get up, causes, prevention.
- https://www.gov.uk/government/news/report-fraud-new-service-from-city-of-london-police : Report Fraud replaced Action Fraud; 0300 123 2040 and Police Scotland 101 came from search results on the same topic (https://www.warwickshire.gov.uk/news/article/7248/report-fraud-has-replaced-action-fraud, https://www.techuk.org/insights/news/report-fraud-service-replaces-action-fraud-as-uk-s-official-fraud-reporting-portal).
- https://stopscamsuk.org.uk/159 : 159 connects to the bank.
- https://consumer.ftc.gov/consumer-alerts/2025/08/business-and-government-impersonators-go-after-older-adults-life-savings : how impersonation scams work, what to do, ReportFraud.ftc.gov.
- https://www.ftc.gov/system/files/ftc_gov/pdf/2026-older-adults-report-10192026.pdf : FTC October 2026 report: investment scams top losses (via search summary; the line was cut from the final text).
- https://files.consumerfinance.gov/f/documents/cfpb_preventing-elder-financial-abuse_friends-family-guide.pdf : family-side warning signs (via search summary; PDF text did not extract).
- https://www.nhs.uk/conditions/social-care-and-support-guide/support-and-benefits-for-carers/carer-assessments/ : carer's assessment.
- https://www.gov.uk/power-of-attorney : LPA types, capacity, registration (England and Wales).
- https://www.nhs.uk/conditions/end-of-life-care/planning-ahead/advance-decision-to-refuse-treatment/ : advance decision requirements.
- https://eurocarers.org/about-carers/ : informal carers in Europe, country profiles.
- https://www.carersuk.org/help-and-advice/your-health-and-wellbeing/ : burnout and looking after yourself.

### Purpose and hard conversations
- https://designingyour.life/wp-content/uploads/2025/03/Odysseys-Worksheet-1.pdf : the three lives, gauges, title each one.
- https://designingyour.life/worksheets-and-discussion-questions/ : dashboard, Good Time Journal, failure log (via search summary).
- https://designingyour.life/insights/the-magic-of-odysseys-prototyping-your-future-with-designing-your-life/ : prototypes through conversations and small experiments (via search summary).
- https://designingyour.life/insights/contemplating-difficult-life-choices-in-the-ai-age-of-immediate-answers/ : no single right plan.
- https://www.beyondintractability.org/bksum/stone-difficult : Difficult Conversations: three conversations, And Stance, intent and impact, contribution, third story, whether to raise it (summary, not the book).
- https://baynvc.org/basics-of-nonviolent-communication/ and https://nonviolentcommunication.com/learn-nonviolent-communication/nvc-happiness/ : NVC four components, requests and demands, interpretations versus feelings (via search summary).
- https://www.nhs.uk/live-well/getting-help-for-domestic-violence/ : signs of control, UK helplines for women, men, LGBT+, device monitoring.
- https://www.nationaldahelpline.org.uk/ : Refuge helpline number, safer device advice.
- https://www.thehotline.org/ : US hotline number, device monitoring.
- https://wave-network.org/find-help/ : national women's helplines across Europe.

### Careers
- https://eur-lex.europa.eu/eli/dir/2023/970/oj/eng : Directive 2023/970 article headings (Articles 5, 7, 34) only; the fetch did not return full text.

### Learning, children, integrity
- https://www.gov.uk/government/publications/generative-ai-product-safety-standards/generative-ai-product-safety-standards : DfE standards, updated 19 January 2026: hints first, no flattery, no personhood, distress and safeguarding handling.
- https://learning.nspcc.org.uk/child-abuse-and-neglect/recognising-and-responding-to-abuse : responding to a disclosure: no promise of secrecy, no in-depth questioning, not their fault, Childline.
- https://childhelplineinternational.org/helplines/ : child helplines by country, 116 111 in several European countries.
- https://arxiv.org/abs/2409.09047 : Lehmann, Cornelius and Sting: substituting versus complementing with a model, widening gap.
- https://academicintegrity.org/resources/fundamental-values : ICAI six values.
- https://www.legislation.gov.uk/ukpga/2022/21/section/26 : England's contract cheating definitions (background, not in final text).
- https://www.psychologicalscience.org/news/releases/which-study-strategies-make-the-grade.html : Dunlosky et al. 2013 ratings (via search summary).
- https://www.evullab.org/pdf/CepedaVulRohrerWixtedPashler-PS-2008.pdf : Cepeda et al. 2008 gap as a share of test delay (via search summary).
- https://psychology.ecu.edu/wp-content/pv-uploads/sites/216/2019/03/Roediger-Karpicke-2006.pdf : testing effect, 5-minute versus delayed test (via search summary).
- https://europass.europa.eu/en/common-european-framework-reference-language-skills : CEFR self-assessment grid (the Council of Europe's own pages returned 403).
- https://en.wikipedia.org/wiki/Common_European_Framework_of_Reference_for_Languages : global scale, Companion Volume 2020 additions.

### Creative
- https://kdp.amazon.com/en_US/help/topic/G200672390 : KDP disclosure of AI-generated versus AI-assisted content.
- https://www.copyright.gov/ai/ : Copyright Office Part 2 date, 29 January 2025 (the page does not state the conclusions).
- https://www.dndbeyond.com/srd : SRD 5.2 under CC BY 4.0.
- https://docs.google.com/document/d/1SB0jsx34bWHZWbnNIVVuMjhDkrdFGo1_hSC2BWPlI3A/ : X-Card by John Stavropoulos (title only).
- https://www.montecookgames.com/consent-in-gaming/ : Consent in Gaming exists and is free.

### Written from memory, not checked against a fetched source
- Lally et al. 2010 (median 66 days, range 18 to 254, a missed day does not derail): UCL and MDPI pages returned 403.
- Lighter training week every 4 to 8 weeks: the paper redirected to a login. Worded in the text as common practice.
- "10 percent a week" as a rule of thumb, not proof: my attempt at the trial's PubMed record hit the wrong paper.
- Sleep restriction cautions (bipolar disorder, epilepsy, driving).
- Neither argue with nor confirm unusual beliefs: Mind's page returned 403; the APA and NHS pages support the urgency, not this exact wording.
- Couples counselling not advised where there is abuse: the Hotline's page returned 404.
- Article 5 and 7 content beyond the headings, and the June 2026 transposition date in Directive 2023/970. National status was not checked; the text tells the agent to check.
- CV norms by country; US salary history bans; unlawful interview questions.
- US list of 9 allergens; 116 117 as a medical on-call number in several EU countries; grapefruit and vitamin K interactions.
- Designing Your Life details: gravity problems, gauge names, reframes, five mindsets.
- Lines and veils from Ron Edwards's Sex and Sorcery (2003); TTRPG Safety Toolkit authors; three clue rule; 2d6 result bands; Lasers & Feelings, Fate Accelerated, Cairn, Ironsworn (site returned 429), Mythic Game Master Emulator.
- Copyright Office's conclusion on human authorship; fiction craft points; editing levels.
- Dunlosky's moderate ratings, interleaving findings, worked-example fading.
- My own rules of thumb, not sourced facts: plan on 80 percent of stated hours, about five focused study hours a day, "roughly weekly" review for a test a month away, review intervals for language mistakes.

---

## Batch: regulated


### Banking: credit
- https://www.eba.europa.eu/regulation-and-policy/credit-risk/guidelines-on-loan-origination-and-monitoring : EBA/GL/2020/06 in force, applying from 30 June 2021.
- https://www.eba.europa.eu/sites/default/files/document_library/Publications/Guidelines/2020/Guidelines%20on%20loan%20origination%20and%20monitoring/884283/EBA%20GL%202020%2006%20Final%20Report%20on%20GL%20on%20loan%20origination%20and%20monitoring.pdf : section numbers, paras 151, 153, 155, 156 to 158 (200 bp example), collateral not a predominant criterion, Annex 3 metrics (read from the PDF text).
- https://www.occ.gov/news-issuances/news-releases/2025/nr-ia-2025-119.html : OCC and FDIC rescinded the 2013 leveraged lending guidance on 5 Dec 2025.
- https://milbank.com/en/news/european-central-bank-ecb-publishes-final-guidance-on-leveraged-lending-transactions.html : ECB 2017 guidance, 4.0x and 6.0x levels (law-firm summary, search snippet only, not the ECB text).
- https://www.whitecase.com/sites/default/files/files/download/publications/ecb-guidance-on-leveraged-transactions_0.pdf : same ECB levels (law-firm summary, search snippet only).

### Banking: AML and sanctions
- https://lawplayer.com/eu/act/32024R1624 : full text of Regulation (EU) 2024/1624 (mirror, EUR-Lex returned empty): Arts. 2(2), 11, 19(4), 20(1)(d), 34, 42, 45(2), 51, 52, 69, 71, 73, 77, 80, 90.
- https://www.cssf.lu/en/2026/10/amla-adopts-key-regulatory-technical-standards/ : AMLA's three draft RTS sent to the Commission, Commission may amend, six-month transition proposed.
- https://www.regulationtomorrow.com/2026/10/amla-finalises-key-standards-for-the-private-sector/ : content and 1 Oct 2026 date of AMLA's final reports.
- https://www.gov.uk/guidance/moving-to-a-single-list-for-uk-sanctions-designations-28-january-2026 : UK Sanctions List the only source since 28 Jan 2026; OFSI list no longer updated.
- https://www.skadden.com/insights/publications/2024/09/eu-updates-sanctions-best-practices : EU Best Practices July 2024, 50% or more, aggregation, control red flags (law-firm summary).
- https://ofac.treasury.gov/faqs/398 : OFAC 50 Percent Rule speaks to ownership only, aggregate.
- https://www.learnsignal.com/blog/ownership-control-sanctions-ofac-50-percent-rule-vs-uk-ofsi-accountants-guide/ : UK more than 50% or control (secondary, search snippet only).
- https://www.legislation.gov.uk/ukpga/2002/29/section/333A : UK tipping-off offence.
- https://fiaumalta.org/?p=16701 : FATF June 2026 public statements (three-times-a-year cycle; search snippet only).
- https://cm.twobirds.com/en/insights/2026/eu-list-of-highrisk-countries-and-jurisdictions-for-money-laundering-and-terrorism-financing-amended : EU high-risk list is a separate delegated act (search snippet only).

### Insurance
- https://www.legislation.gov.uk/eudr/2016/97/article/2 : IDD Art. 2(2)(b), claims management and loss adjusting outside distribution.
- https://www.legislation.gov.uk/eudr/2016/97/article/20 : IDD Art. 20(1) to (5).
- https://www.legislation.gov.uk/eudr/2016/97/article/22 : IDD Art. 22(1), large risks.
- https://www.eiopa.europa.eu/qa-regulation/questions-and-answers-database/2525-signature-requirements-relating-demands-and-needs-statement_en : no signature or format requirement; applies with or without advice; not a suitability assessment (search snippet only).
- https://www.eiopa.europa.eu/qa-regulation/questions-and-answers-database/2694-format-demands-and-needs-test-documentation-case-large-risks_en : large risks and stricter national rules.
- https://eur-lex.europa.eu/EN/legal-content/summary/civil-liability-insurance-for-motor-vehicles.html : Directive 2009/103 Art. 22, three-month reasoned offer (search snippet only).
- https://handbook.fca.org.uk/handbook/ICOBS/8/2.html : ICOBS 8.2 motor (search snippet only; ICOBS 8.1.1R also from search snippets, the handbook page itself returned no text).
- https://www.legislation.gov.uk/ukpga/2015/4/section/3 : UK duty of fair presentation.
- https://www.legislation.gov.uk/ukpga/2015/4/section/11 : breach unrelated to the actual loss.
- https://www.legislation.gov.uk/ukpga/2015/4/section/13A : payment within a reasonable time.
- https://www.gesetze-im-internet.de/vvg_2008/__14.html : VVG s.14 due date and advance payment.

### Healthcare: coding and billing
- https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/ICD10CM/2027/ICD-10-CM-October-1-2026-FY27-Guidelines.pdf : FY 2027 validity dates, I.A.19, II.H, IV.H (read from the PDF text).
- https://www.cms.gov/files/document/mln1783722-proper-use-modifiers-59-xe-xp-xs-xu.pdf : indicators 0 and 1, modifier 59 and X{EPSU} use, not on E/M, modifier 25 (MLN April 2026, read from the PDF text).
- https://www.cms.gov/medicare/coding/ncci-coding-edits : PTP and MUE purpose, quarterly updates.
- https://www.who.int/standards/classifications/frequently-asked-questions/icd-11-implementation : ICD-11 effective 1 Jan 2022, ICD-10 usable as long as necessary, no penalty.
- https://www.bfarm.de/EN/Code-systems/Classifications/ICD/ICD-10-GM/_node.html : ICD-10-GM 2026 current, valid by calendar year.
- https://acdis.org/resources/guidelines-achieving-compliant-query-practice%E2%80%942022-update : AHIMA/ACDIS query brief, 2022 and 2023 addendum (announcements, search snippet only).

### Healthcare: prior authorization
- https://www.cms.gov/newsroom/fact-sheets/cms-interoperability-prior-authorization-final-rule-cms-0057-f : CMS-0057-F payers, 72 hours and 7 days, denial reasons, metrics by 31 March, API 1 Jan 2027, drugs excluded.
- https://www.law.cornell.edu/cfr/text/29/2560.503-1 : ERISA 180 days, 72 hours, 15 and 30 days, clinician consult, notice content.
- https://www.law.cornell.edu/cfr/text/42/422.582 : Medicare Advantage reconsideration, 60 days, 5-day receipt presumption.
- https://www.everycrsreport.com/files/2026-05-12_IF13133_1b7af5c4cb1305bfa031b37cfcf21ca5739edb1a.html : WISeR six states, began 1 Jan 2026 (search snippet only).
- https://medicare.chir.georgetown.edu/?p=985 : WISeR faces potential repeal (headline only, hence "check").
- https://www.gesetze-im-internet.de/sgb_5/__13.html : SGB V s.13(3a) three and five weeks, deemed approval.

### Healthcare: scribe, privacy
- https://www.law.cornell.edu/cfr/text/45/164.502 : HIPAA minimum necessary and treatment exception (HHS page returned 403).
- https://gdpr-info.eu/art-9-gdpr/ : GDPR Art. 9(1), 9(2)(h), 9(3).
- https://www.gesetze-im-internet.de/stgb/__201.html : StGB s.201.
- https://www.paubox.com/blog/does-an-ai-clinical-scribe-need-to-announce-that-it-is-recording : recording consent comes from state law, not HIPAA (secondary, search snippet only).
- https://www.gfrlaw.com/what-we-do/insights/ambient-scribes : a business associate agreement does not settle recording law (law-firm note, search snippet only).

### Life sciences: evidence and regulatory writing
- https://www.cochrane.org/authors/handbooks-and-manuals/handbook/current : Handbook version 6.5 (2024) listed as current.
- https://www.cochrane.org/zh-hant/authors/handbooks-and-manuals/handbook/current/chapter-14 : GRADE levels, five domains, summary of findings (search snippet only).
- https://www.ema.europa.eu/en/ich-e3-structure-content-clinical-study-reports-scientific-guideline : E3 current version from 1996, with Q&A (R1), no revision listed.
- https://www.ema.europa.eu/en/ich-e6-good-clinical-practice-scientific-guideline : E6(R3) 23 July 2025, Annex 2 and consolidated text 15 Jan 2027.
- https://www.ema.europa.eu/en/ich-m11-guideline-clinical-study-protocol-template-technical-specifications-scientific-guideline : M11 Step 5, first published 15 Dec 2025, no date of effect given.
- https://lawplayer.com/eu/act/32014R0536 : EU CTR Art. 37(4) (mirror).
- https://lawplayer.com/eu/act/32017R0745 : MDR Art. 61(1) to (6) and (10) to (12) (mirror).
- https://health.ec.europa.eu/medical-devices-sector/new-regulations/guidance-mdcg-endorsed-documents-and-other-guidance_en : MDCG list and dates; no 2025 or 2026 revision of the core clinical documents.
- https://www.emergobyul.com/news/europe-targets-revised-mdr-and-ivdr-adoption-q2-q3-2027 : revision status, Parliament vote Q1 2027, adoption mid-2027, current MDR stays applicable.

### Real estate
- https://www.law.cornell.edu/uscode/text/42/3604 : 3604(c) and protected classes.
- https://www.law.cornell.edu/cfr/text/24/100.75 : written and oral notices, images, choice of media.
- https://www.nclc.org/resources/fact-sheet-hud-rule-would-gut-housing-discrimination-protections/ : HUD disparate-impact proposal of 14 Jan 2026, not final per this source.
- https://lawplayer.com/eu/act/32000L0043 : Directive 2000/43 Arts. 3(1)(h), 8 (mirror).
- https://www.gesetze-im-internet.de/agg/__19.html : AGG s.19 grounds and 50-flat threshold.
- https://www.legislation.gov.uk/ukpga/2010/15/section/33 : Equality Act s.33.
- https://www.gov.uk/government/publications/renters-rights-act-2025-implementation-roadmap/implementing-the-renters-rights-act-2025-our-roadmap-for-reforming-the-private-rented-sector : 1 May 2026 changes, database from late 2026, Awaab's Law for private rented pending.
- https://www.gov.uk/government/publications/guide-to-the-renters-rights-act/guide-to-the-renters-rights-act : section 13, 3-month arrears ground, Awaab's Law in social housing from 27 Oct 2025.
- https://lawplayer.com/eu/act/32024L1275 : EPBD recast Arts. 20(2), 20(4), 22(2), 35 (mirror).
- https://www.gesetze-im-internet.de/bgb/__556.html : BGB s.556(3).
- https://www.gesetze-im-internet.de/bgb/__578.html : BGB s.578, text form via s.550.
- https://kpmg.com/us/en/articles/2022/lease-accounting-ifrs-standards-us-gaap.html : IFRS 16 vs ASC 842 on model, index remeasurement, low-value, short-term, risk-free rate.
- https://schneiderdowns.com/our-thoughts-on/fasb-asc-topic-842-vs-ifrs-16/ : same, plus borrowing rate wording.
- https://www.frc.org.uk/library/standards-codes-policy/accounting-and-reporting/uk-accounting-standards/frs-102/ : Periodic Review 2024, effective 1 Jan 2026 (the page does not mention leases).

### Not confirmed live
The search allowance ran out during research, so these rest on established knowledge and were not re-read today. The ones most worth a second look come first.
- California AB 723 (altered listing photos) and Penal Code 632 (all-party consent): official site returned 403.
- The ban on upward-only rent reviews in England: the bill's page returned 403, status unknown. The text tells the agent to check.
- FRS 102 lease change: the FRC page confirms the effective date, not the lease content.
- SGB V s.13(3a) narrowing by the courts (BSG 2020 case law): recalled, not re-read.
- FDA position on E6(R3) Annex 2: secondary sources disagreed, so the text says unconfirmed.
- NHS England ambient scribing guidance: page returned empty. Nothing in the text cites it.
- From established knowledge: AMLD4 Arts. 3(6), 39; FATF R.10; IDD Arts. 17, 19, 23; Directives 2004/113, 93/13 Art. 5; ICOBS 5.2, 8.1.2R; VVG ss.19, 28, 81; GDPR Art. 22; CJEU C-634/21; AI Act Annex III(5)(b); NCCI indicator 9 and MUE practice; 60-day overpayment duty; 42 CFR 422.101(b); 45 CFR 147.136; SGG s.84; German outpatient certainty markers; UK and French code systems; ISMP and Joint Commission lists; GRADE language and rating rules; RoB 2, ROBINS-I, AMSTAR 2, QUADAS-2, AGREE II, ICEMAN; ICH E6(R3) Appendix B, E9(R1), E3 section numbers; MDCG document subjects; IFRS 16 paras 18, 19, B37; Landlord and Tenant Act 1954; DMCC Act 2024 and NTSELAT guidance; BGB s.551; UK gas and electrical check cycles; national price data sources.
