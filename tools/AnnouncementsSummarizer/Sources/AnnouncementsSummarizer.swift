import Foundation
import FoundationModels

/// Summarizes `announcements.txt` using the Foundation Models framework (Apple Intelligence).
/// Heavier requests may be handled by the system; routing to Private Cloud Compute is not app-controllable.
@main
enum AnnouncementsSummarizer {
    private static let maxChunkChars = 5_500

    static func main() async {
        do {
            try await run()
        } catch {
            let msg = String(describing: error)
            FileHandle.standardError.write(Data("announcements-summarizer: \(msg)\n".utf8))
            exit(1)
        }
    }

    private static func run() async throws {
        let args = CommandLine.arguments.dropFirst()
        guard let pathArg = args.first, pathArg != "-h", pathArg != "--help" else {
            FileHandle.standardError.write(
                Data("usage: announcements-summarizer <path-to-announcements.txt>\n".utf8)
            )
            exit(2)
        }

        let fileURL = URL(fileURLWithPath: pathArg, isDirectory: false)
        let data = try Data(contentsOf: fileURL)
        guard !data.isEmpty else {
            FileHandle.standardError.write(
                Data("Input file is empty; nothing to summarize.\n".utf8)
            )
            exit(1)
        }
        let raw = String(data: data, encoding: .utf8)
            ?? String(decoding: data, as: UTF8.self)

        let model = SystemLanguageModel.default
        if model.availability != .available {
            FileHandle.standardError.write(
                Data(
                    "Foundation model is not available. Enable Apple Intelligence in System Settings (Apple Silicon + supported macOS required).\n"
                        .utf8
                )
            )
            exit(1)
        }

        let summary = try await summarizeLongText(raw)
        print(summary, terminator: "")
    }

    private static func summarizeLongText(_ text: String) async throws -> String {
        if text.count <= maxChunkChars {
            return try await summarizeChunk(text)
        }

        // Split on announcement separators first, then by size.
        var pieces: [String] = []
        for block in text.components(separatedBy: "\n---\n") {
            var remaining = block.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !remaining.isEmpty else { continue }
            while !remaining.isEmpty {
                if remaining.count <= maxChunkChars {
                    pieces.append(remaining)
                    break
                }
                let endIdx = remaining.index(remaining.startIndex, offsetBy: maxChunkChars)
                var cut = String(remaining[..<endIdx])
                if let lastNewline = cut.lastIndex(of: "\n") {
                    cut = String(remaining[..<lastNewline])
                }
                if cut.isEmpty {
                    cut = String(remaining.prefix(maxChunkChars))
                }
                pieces.append(cut)
                remaining = String(remaining[cut.endIndex...]).trimmingCharacters(in: .whitespacesAndNewlines)
            }
        }

        if pieces.isEmpty { return try await summarizeChunk(text) }
        if pieces.count == 1 { return try await summarizeChunk(pieces[0]) }

        var partSummaries: [String] = []
        for (i, p) in pieces.enumerated() {
            let s = try await summarizeChunk(
                p,
                instructionExtra: "This is part \(i + 1) of \(pieces.count) of a longer log. Capture names, dates, and product facts."
            )
            partSummaries.append(s)
        }
        return try await mergePartSummaries(partSummaries)
    }

    private static func summarizeChunk(
        _ text: String,
        instructionExtra: String = ""
    ) async throws -> String {
        let extra = instructionExtra.isEmpty ? "" : " \(instructionExtra)"
        let instructions = """
        You condense long announcement logs for quick scanning. Use markdown: short bullets, optional ## headings. \
        Keep proper nouns, version numbers, and dates. No preamble or closing remarks.\(extra)
        """
        let session = LanguageModelSession(instructions: instructions)
        let prompt = """
        Summarize this text (announcements from social/creator feeds, possibly noisy):

        \(text)
        """
        return try await respondString(session, prompt: prompt)
    }

    private static func mergePartSummaries(_ parts: [String]) async throws -> String {
        let instructions = """
        You merge several partial summaries of the same running announcement log. \
        Deduplicate repeated facts. One coherent markdown output (bullets and ## as needed). No preamble.
        """
        let session = LanguageModelSession(instructions: instructions)
        let joined = parts.enumerated()
            .map { idx, s in
                "### Part \(idx + 1)\n\n\(s)"
            }
            .joined(separator: "\n\n")
        let prompt = """
        Merge and deduplicate into one skimmable summary:

        \(joined)
        """
        return try await respondString(session, prompt: prompt)
    }

    private static func respondString(_ session: LanguageModelSession, prompt: String) async throws
        -> String
    {
        do {
            let response = try await session.respond(to: prompt)
            return String(describing: response.content)
        } catch let error as LanguageModelSession.GenerationError {
            if case .exceededContextWindowSize = error {
                // Last resort: hard-split prompt body (rare for our chunk size).
                let half = prompt.count / 2
                let i = prompt.index(prompt.startIndex, offsetBy: half)
                let a = String(prompt[..<i])
                let b = String(prompt[i...])
                let s1 = try await summarizeChunk(a, instructionExtra: "Fragment A of oversized chunk.")
                let s2 = try await summarizeChunk(b, instructionExtra: "Fragment B of oversized chunk.")
                return try await mergePartSummaries([s1, s2])
            }
            throw error
        }
    }
}
