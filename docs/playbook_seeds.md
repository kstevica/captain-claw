# Playbook Seeds for Wizard

Copy-paste each block into the playbook wizard. They're specific enough to minimize follow-up questions.

---

## WEB RESEARCH

### 1
I want a playbook for researching a company before a sales call. Given a company name and domain, search the web for recent news, funding rounds, key people, and product launches. Fetch the top 5 results one at a time, extract key findings into individual markdown files, then produce a single company brief. Use web_search → web_fetch → write per result, then combine. Output: company-brief.md with sections for Overview, Recent News, Key People, Products, Funding.

### 2
I want a playbook for monitoring competitor pricing. Given a list of competitor URLs (product/pricing pages), fetch each page one at a time with web_fetch, extract pricing tiers and features into a structured markdown table, write each to a file, then produce a comparison matrix. Flag any changes from the previous run if a prior report exists.

### 3
I want a playbook for finding and summarizing academic papers on a topic. Use web_search to find 10 relevant papers/articles, fetch each one at a time with web_fetch, extract title/authors/abstract/key findings, write per-paper summary files, then produce a literature review markdown with a summary table and key themes.

### 4
I want a playbook for tracking regulatory changes in a specific industry. Search for recent regulatory updates using web_search with date-focused queries, fetch the top 8 results one at a time, extract the regulation name/agency/effective date/impact summary, write per-regulation files, then combine into a regulatory update report sorted by effective date.

### 5
I want a playbook for researching job market trends for a specific role. Search for salary surveys, job postings analysis, and market reports using web_search. Fetch top 8 results one at a time, extract salary ranges/required skills/demand trends/remote vs onsite ratios, write per-source summaries, then produce a job market report with aggregated data and trends.

### 6
I want a playbook for building a technology comparison report. Given 3-5 technology names (frameworks, tools, platforms), search for each one's docs/reviews/benchmarks, fetch results one at a time, extract pros/cons/use cases/performance data, then produce a side-by-side comparison report with a recommendation matrix.

### 7
I want a playbook for researching a news event in depth. Given a headline or topic, search for coverage from multiple sources, fetch 10 articles one at a time, extract key facts/quotes/timeline events/different perspectives, write per-source summaries, then synthesize into a comprehensive briefing with a timeline and fact-check notes.

---

## BATCH PROCESSING

### 8
I want a playbook for renaming files in bulk based on their content. Given a directory of files (PDFs, docs, images), read or extract content from each, determine a descriptive filename from the content (date + topic + type), then rename using shell mv commands. Process one at a time. Write a rename log showing old → new names.

### 9
I want a playbook for batch-converting markdown files to a standardized format. Given a directory of .md files, glob to find them all, read each one, normalize headings (ensure H1 exists, fix heading hierarchy), add frontmatter if missing (title, date, tags), fix broken links, then write the cleaned version back. Process one file at a time.

### 10
I want a playbook for extracting and cataloging all TODO/FIXME comments from a codebase. Glob for all source files, read each file, extract lines containing TODO/FIXME/HACK/XXX with file path and line number, categorize by priority/type, then write a consolidated todo-report.md sorted by priority with file references.

### 11
I want a playbook for batch-processing invoices from PDFs. Given a directory of invoice PDFs, use pdf_extract on each one at a time, extract vendor name/invoice number/date/line items/total amount, write structured data per invoice, then produce a summary spreadsheet-style markdown with totals and a vendor breakdown.

### 12
I want a playbook for batch image analysis. Given a directory of product images, use image_vision on each one at a time to extract: product type, color, condition, notable features. Write a per-image description file, then combine into a product catalog markdown with image references and descriptions.

### 13
I want a playbook for processing a folder of meeting recording transcripts. Use summarize_files for the initial pass, then for each transcript extract: attendees, action items, decisions made, follow-up dates. Write per-meeting summary files, then produce a combined action-item tracker sorted by due date and assignee.

### 14
I want a playbook for batch OCR processing of scanned documents. Given a directory of scanned images/PDFs, use image_ocr on each one at a time, clean up the extracted text (fix common OCR errors), write the clean text to corresponding .txt files, then produce an index file listing all documents with their extracted titles and dates.

### 15
I want a playbook for validating and fixing CSV data files in batch. Given a directory of CSVs, read each one, check for: missing required columns, data type mismatches, duplicate rows, empty required fields. Write a validation report per file, fix what can be auto-fixed, then produce a summary of all issues found and fixed.

---

## CODE GENERATION

### 16
I want a playbook for generating REST API endpoint boilerplate. Given an API spec (resource name, fields, relationships), generate: route handler, request/response models, validation logic, database query functions, and basic tests. Follow the existing project patterns by reading 2-3 similar existing endpoints first. Write each file and run the test suite.

### 17
I want a playbook for generating database migration scripts. Given a description of schema changes (new tables, altered columns, new indexes), read the existing schema/migrations, generate an up and down migration following the project's migration framework conventions, write the migration file, then run a dry-run validation.

### 18
I want a playbook for scaffolding a new microservice. Given a service name and purpose, generate: Dockerfile, docker-compose entry, config files, health check endpoint, logging setup, basic CI pipeline config. Read existing services first to match patterns. Write all files, then validate with a build command.

### 19
I want a playbook for generating unit tests for existing functions. Given a source file or directory, read the code, identify all public functions/methods, generate test cases covering: happy path, edge cases, error conditions, boundary values. Follow existing test patterns in the project. Write test files and run them to verify they pass.

### 20
I want a playbook for generating TypeScript types from JSON examples. Given a directory of JSON sample files, read each one, infer TypeScript interfaces with proper types (handling nullable, arrays, nested objects), generate a types.ts file with all interfaces, and optionally generate Zod validation schemas that match.

### 21
I want a playbook for generating API client code from endpoint definitions. Read the apis tool for registered endpoints, generate typed client functions for each (with proper request/response types, error handling, retry logic), write to a client library file, and generate basic integration tests.

### 22
I want a playbook for refactoring repeated code patterns. Given a codebase directory, glob for source files, read them, identify repeated code blocks (3+ occurrences of similar logic), propose extracted functions/utilities, generate the shared module, then edit each file to use the new shared code. Run tests after each edit.

---

## DOCUMENT PROCESSING

### 23
I want a playbook for converting a collection of documents into a knowledge base. Given a directory with mixed formats (PDF, DOCX, MD, TXT), use summarize_files for initial processing, then for each document extract: title, category, key topics, summary. Write a structured index.md with categories and cross-references, plus individual summary pages.

### 24
I want a playbook for extracting structured data from contracts/legal documents. Given PDF contracts, use pdf_extract on each one at a time, identify: parties, effective dates, term length, key obligations, termination clauses, payment terms. Write structured summary per contract, then produce a contract comparison table.

### 25
I want a playbook for creating an executive summary from a large report. Given a long document (PDF or DOCX), extract its content, identify the main sections and key findings, generate a 1-2 page executive summary with: key metrics, major findings, recommendations, and action items. Write as both markdown and prepare for DOCX export.

### 26
I want a playbook for merging and deduplicating contact lists from multiple sources. Given multiple files (CSV, XLSX, vCard) containing contact information, extract data from each, normalize fields (phone formats, email case, name capitalization), identify duplicates by fuzzy matching on name+email, merge records keeping the most complete data, write a clean consolidated contact list.

### 27
I want a playbook for processing expense reports from receipts. Given a folder of receipt images and PDFs, use image_ocr and pdf_extract on each one at a time, extract: vendor, date, amount, category, payment method. Write per-receipt data files, then produce a combined expense report with category totals, date range, and flagged duplicates.

### 28
I want a playbook for creating a changelog from git history between two tags/dates. Use shell to run git log with the right format, parse commit messages, categorize into: features, fixes, breaking changes, docs, refactors. Generate a CHANGELOG.md following Keep a Changelog format with proper sections and links.

---

## DATA TRANSFORMATION

### 29
I want a playbook for transforming API response data into report format. Given a set of API endpoint definitions (in the apis tool), call each endpoint via direct_api, collect the JSON responses, transform the data by: flattening nested objects, computing aggregates, formatting dates/currencies, then write a formatted markdown report with tables.

### 30
I want a playbook for normalizing inconsistent data across multiple files. Given CSV/JSON files with the same logical data but different schemas (different column names, date formats, units), read each file, map columns to a canonical schema, convert formats, then write a single normalized output file with a mapping log.

### 31
I want a playbook for generating charts/visualizations data from raw datasets. Given CSV or JSON data files, read and analyze the data, identify interesting patterns/trends/outliers, generate Python matplotlib/plotly scripts that create visualizations, run them via shell to produce image files, then write a report embedding the chart descriptions and key insights.

### 32
I want a playbook for ETL from Google Drive spreadsheets. Search Google Drive for specific spreadsheets by name/folder, read their contents, transform the data (clean, validate, compute new columns), then write the results to either local files or back to Google Drive in a different format/structure. Log all transformations applied.

### 33
I want a playbook for converting between data formats in bulk. Given a directory of files in one format (JSON, CSV, XML, YAML), glob to find them all, read each one at a time, convert to the target format preserving all data, write the converted file alongside the original, then produce a conversion report with any issues encountered.

### 34
I want a playbook for aggregating metrics from multiple log files. Given a directory of application log files, read each one at a time, extract timestamps/error codes/response times/user IDs, compute aggregates (error rates, p50/p95/p99 latency, top errors, busiest hours), write the analysis per log file, then produce a combined metrics dashboard in markdown.

---

## ORCHESTRATION

### 35
I want a playbook for the daily standup preparation workflow. Every morning: check Google Calendar for today's meetings, search Gmail for unread messages from the team, check todo items for overdue tasks, read any recent Google Drive docs shared with me, then produce a morning briefing markdown with: today's schedule, pending items, messages needing response, and suggested priorities.

### 36
I want a playbook for the weekly report generation workflow. At end of week: review completed todo items, check git log for commits this week (via shell), search Gmail for important threads, check Google Calendar for meetings attended, then produce a weekly summary report with: accomplishments, blockers, next week's plan, key decisions made.

### 37
I want a playbook for onboarding a new project. Given a git repository URL: clone it via shell, glob for README/docs/config files, read key files to understand structure, identify the tech stack and dependencies, run any setup commands, then produce an onboarding guide with: project overview, architecture, setup steps, key files, and development workflow.

### 38
I want a playbook for incident response documentation. When an incident occurs: gather the timeline from chat/email (Gmail search), collect relevant logs (read log files), identify affected systems, document the root cause analysis, list remediation steps taken, and produce a post-mortem document following the standard template with: summary, impact, timeline, root cause, action items.

### 39
I want a playbook for multi-step data pipeline orchestration. Given a pipeline definition (source → transform → validate → load), execute each step sequentially, validate output at each stage (row counts, schema checks, null checks), write checkpoint files between stages, and produce a pipeline execution report with timing and validation results. If any stage fails, stop and report.

### 40
I want a playbook for content publishing workflow. Given a draft document: check spelling/grammar, verify all links work (web_fetch each link), optimize images if present, generate social media snippets (tweet-length, LinkedIn-length), create an email newsletter version, and write all outputs to a publish/ directory ready for distribution.

### 41
I want a playbook for coordinating a multi-agent research task. Given a complex research question, decompose it into 3-5 sub-questions, use botport to delegate each sub-question to a specialist agent, collect results, identify contradictions or gaps, synthesize into a coherent answer, and produce a research report with source attribution.

---

## FILE MANAGEMENT

### 42
I want a playbook for organizing a messy downloads folder. Given a directory path, glob for all files, categorize by: file type (documents, images, videos, archives, code, data), date (this week, this month, older), size (large >100MB, medium, small). Create subdirectories by category, move files using shell mv, write an organization report showing what went where.

### 43
I want a playbook for finding and cleaning duplicate files. Given a directory, glob recursively for all files, compute checksums via shell (md5/sha256), identify exact duplicates, group them, keep the one with the best name/newest date, list the rest for deletion, write a report showing space savings. Do NOT auto-delete — produce a cleanup script instead.

### 44
I want a playbook for archiving old project files. Given a project directory and an age threshold, glob for all files, check modification dates via shell, identify files older than the threshold, create a compressed archive via shell (tar/zip), move archived files, update any references in config/docs, and produce an archive manifest.

### 45
I want a playbook for syncing file structures between two directories. Given source and destination paths, glob both directories, compare file lists and modification times, identify: new files, modified files, deleted files. Produce a sync plan showing all changes, then execute the sync using shell cp/rm commands after confirmation. Write a sync log.

### 46
I want a playbook for generating a project documentation site from source files. Glob for all markdown files in a docs/ directory, read each to extract title and headings, generate a table of contents with proper nesting, create navigation links between pages, generate an index page, and write a _sidebar.md for documentation tools like Docsify.

---

## INTERACTIVE

### 47
I want a playbook for guided interview/questionnaire processing. Given a questionnaire template (list of questions), present each question to the user one at a time, record their answers, process answers into a structured report, identify any inconsistencies or missing information, then produce a final formatted document with all responses organized by section.

### 48
I want a playbook for interactive code review assistance. Given a PR diff or changed files, read each changed file, analyze for: bugs, security issues, performance problems, style violations, missing tests. Present findings one at a time to the user, let them accept/dismiss/modify each finding, then produce a review summary with accepted items as actionable feedback.

### 49
I want a playbook for interactive data exploration. Given a dataset (CSV/JSON), read and profile it (row count, columns, types, distributions, nulls, unique values), present the profile to the user, ask what questions they want to answer, then iteratively: write analysis scripts, run them, present results, ask for next direction. Build up an analysis notebook.

### 50
I want a playbook for guided budget planning. Ask the user for income sources and amounts, then walk through expense categories one at a time (housing, food, transport, utilities, entertainment, savings, etc.), record amounts, compute totals and ratios, compare against standard budgeting rules (50/30/20), then produce a budget plan with recommendations and a monthly tracking template.

---

## MIXED / ADVANCED

### 51
I want a playbook for SEO audit of a website. Given a domain, fetch the homepage with web_fetch, extract meta tags/headings/links, check for common SEO issues (missing meta descriptions, broken links, missing alt text, heading hierarchy). Fetch 5 key pages one at a time and repeat the analysis. Produce an SEO audit report with prioritized fixes.

### 52
I want a playbook for email campaign preparation. Given a topic and audience segment: search contacts for matching recipients, draft the email content, generate subject line variants, create a plain-text and HTML version, write all variants to files for review, and produce a send-ready package with recipient list, content, and a send checklist.

### 53
I want a playbook for competitive feature analysis. Given our product's feature list (from a file) and 3-5 competitor names, search the web for each competitor's features/pricing, fetch their product pages one at a time, extract features, build a feature comparison matrix, identify our gaps and advantages, then produce a competitive analysis report with strategic recommendations.

### 54
I want a playbook for codebase health check. Given a project directory: count lines of code by language (shell cloc/wc), check for outdated dependencies (shell npm outdated/pip list --outdated), find TODO/FIXME counts, measure test coverage if possible, check for large files, find unused imports, then produce a health report card with scores and recommendations.

### 55
I want a playbook for meeting preparation. Given a meeting topic and attendee list: search Gmail for recent threads with attendees, check Google Calendar for past meetings with them, search Google Drive for shared documents, search the web for any relevant external context, then produce a meeting prep brief with: attendee background, open threads, relevant docs, suggested agenda, and talking points.

### 56
I want a playbook for automated changelog and release notes generation. Given a version tag or date range: run git log via shell, categorize commits (features/fixes/docs/chores), fetch linked issue details if any URLs are in commit messages (web_fetch one at a time), generate user-facing release notes (non-technical language), developer changelog (technical), and update the CHANGELOG.md file.

### 57
I want a playbook for knowledge transfer documentation. Given a codebase directory: identify the key modules (glob + read), trace the main execution flow, document architecture decisions, create a glossary of domain terms found in the code, generate onboarding FAQs, and produce a knowledge transfer package with architecture diagrams (as text), module guides, and a getting-started walkthrough.

### 58
I want a playbook for data quality audit. Given a database or data files: read/extract the data, profile each column (completeness, uniqueness, distribution, format consistency), identify anomalies and outliers, check referential integrity between related datasets, then produce a data quality report with scores per field and a prioritized remediation plan.

### 59
I want a playbook for automated social media content calendar. Given a content strategy doc and posting schedule: read the strategy, generate post ideas for the next 4 weeks following the content pillars, create draft posts for each platform (different lengths/formats), organize into a calendar view (markdown table by week and day), and write all drafts to a content/ directory.

### 60
I want a playbook for API health monitoring report. Given registered API endpoints (from the apis tool), call each one via direct_api, measure response time, check for errors, validate response schema, compare against expected baselines, then produce a health dashboard markdown with: status per endpoint, latency stats, error rates, and any degradations flagged.
