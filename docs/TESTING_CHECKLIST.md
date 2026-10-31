# Flight Deck manual test checklist

[← README](../README.md) · [Getting started](guide/getting-started.md) · [Flight Deck guide](guide/flight-deck.md)

A short manual smoke test for the Flight Deck dashboard. Run it on a clean machine or a fresh virtual environment, top to bottom. Tick each box when the expected result happens; note the step and what you saw for any that don't.

## Install

- [ ] Create a fresh virtual environment and run `pip install captain-claw`. **Expected:** the install finishes with no errors.
- [ ] Run `which flight-deck`. **Expected:** the command is found on your `PATH`.

## Start the dashboard

- [ ] Run `flight-deck` in a terminal. **Expected:** the process stays running and logs that it is serving on port 25080, with no traceback.
- [ ] Open <http://localhost:25080> in a browser. **Expected:** the Flight Deck page loads, with no blank screen and no errors in the browser console.
- [ ] On a first run, check the first screen. **Expected:** onboarding starts.
- [ ] Reload the page. **Expected:** the dashboard loads again and keeps your state.

## Create an agent

- [ ] Use the dashboard's create-agent control and give the agent a name. **Expected:** a creation form or dialog opens.
- [ ] Pick a model and provide a key (or choose a local Ollama model), then confirm. **Expected:** the agent appears on the dashboard and its status becomes running or ready, with no error banner.
- [ ] Refresh the page. **Expected:** the agent is still listed.

## Send a message

- [ ] Open the new agent's chat and send `Hello, what can you do?`. **Expected:** your message appears in the conversation at once.
- [ ] Wait for the reply. **Expected:** the agent answers in the chat, with no error message.
- [ ] Send a second message that refers to the first (for example `Repeat my last question`). **Expected:** the reply shows the agent kept the conversation context.

## Stop the server

- [ ] In the terminal running `flight-deck`, press `Ctrl+C`. **Expected:** the process shuts down cleanly and returns you to the prompt, with no traceback.
- [ ] Reload <http://localhost:25080>. **Expected:** the page no longer loads (connection refused).
- [ ] Run `flight-deck` again and open the dashboard. **Expected:** it starts normally and the agent you created is still there.
