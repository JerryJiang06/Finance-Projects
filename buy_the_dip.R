library(DBI)
library(RSQLite)
library(dplyr)
library(TTR)
library(ggplot2)
library(zoo)

con <- dbConnect(RSQLite::SQLite(), "C:/Users/jerry/quant/etf_data.db") #this dataset uses adj close, which skews historical prices downwards

df <- dbReadTable(con, "etf_prices")

df$Date <- as.Date(df$Date)
df <- df %>%
  arrange(Date) %>%
  select(Date, SPY) %>%
  rename(Close = SPY)

df <- df %>%
  mutate(
    SMA50 = SMA(Close, n = 50),
    SMA200 = SMA(Close, n = 200),
    RSI14 = RSI(Close, n = 30) #30 for longer term holds
  )

#Buy signal when:
#1. 50DMA crosses above 200DMA (golden cross)
#2. RSI is below 80 to avoid overbought (this is mostly to test out RSI function, I don't think this is that useful yet)

#Sell signal when 50DMA crosses under 200DMA (death cross)

rsi_cutoff <- 80

df <- df %>%
  mutate(
    golden_cross = (SMA50 > SMA200) & (lag(SMA50) <= lag(SMA200)),
    buy_signal = golden_cross & (RSI14 < rsi_cutoff),
    death_cross = (SMA50 < SMA200) & (lag(SMA50) >= lag(SMA200)),
    sell_signal = death_cross
  )





# For each buy signal, calculate forward returns over 1, 3, 6, 12 months
# forward_return <- function(x, n) {
#   c(rep(NA, n), (tail(x, -n) / head(x, -n)) - 1)
# }

# df <- df %>%
#   mutate(
#     fwd21 = forward_return(Close, 21),
#     fwd63 = forward_return(Close, 63),
#     fwd126 = forward_return(Close, 126),
#     fwd252 = forward_return(Close, 252)
#   )

# # Only keep buy signal dates
# signals <- df %>% filter(buy_signal == TRUE)
# #print(signals)

# cat("Number of buy signals:", nrow(signals), "\n")
# cat("Average 1-month return:", mean(signals$fwd21, na.rm = TRUE), "\n")
# cat("Average 3-month return:", mean(signals$fwd63, na.rm = TRUE), "\n")
# cat("Average 6-month return:", mean(signals$fwd126, na.rm = TRUE), "\n")
# cat("Average 12-month return:", mean(signals$fwd252, na.rm = TRUE), "\n")

# windows()
# #Plot: SPY with Moving Averages and Buy Signals
# #png(file="SPY_plot.png", width = 1920, height = 1280)
# p1 <- ggplot(df, aes(x = Date)) +
#   geom_line(aes(y = Close, color = "SPY")) +
#   geom_line(aes(y = SMA50, color = "50 DMA"), linewidth = 0.7) +
#   geom_line(aes(y = SMA200, color = "200 DMA"), linewidth = 0.7) +
#   geom_point(data = signals, aes(y = Close), color = "darkgreen", size = 2, shape = 24, fill = "green") +
#   labs(
#     title = "SPY Golden Cross + RSI Buy Signals",
#     y = "Price",
#     x = "Date",
#     color = "Legend"
#   ) +
#   theme_minimal()
# #print(p1)

# #RSI Chart
# p2 <- ggplot(df, aes(x = Date, y = RSI14)) +
#   geom_line(color = "steelblue") +
#   geom_hline(yintercept = rsi_cutoff, color = "red", linetype = "dashed") +
#   labs(
#     title = "SPY RSI (14-Day)",
#     y = "RSI",
#     x = "Date"
#   ) +
#   theme_minimal()
# #print(p2)





df <- df %>%
  mutate(
    Position = NA_real_, #start with all NAs
    Position = ifelse(buy_signal, 1, Position), #make position 1 if buy signal 
    Position = ifelse(sell_signal, 0, Position) #make position 0 if sell signal
  )


df$Position <- zoo::na.locf(df$Position, na.rm = FALSE) #carry forward the last signal
df$Position[is.na(df$Position)] <- 1  #assume in the market at the start

df <- df %>%
  mutate(
    Daily_Return = Close / lag(Close) - 1,
    Strategy_Return = Daily_Return * lag(Position, default = 0)
  )

df[is.na(df)] <- 0 #replace NAs with 0

df <- df %>%
  mutate(
    SPY_Return = cumprod(1 + Daily_Return),
    Strat_Return = cumprod(1 + Strategy_Return),
    Relative_Perf = Strat_Return / SPY_Return
  )

print(head(df))

#Summary Metrics
total_return_spy <- tail(df$SPY_Return, 1) - 1
total_return_strat <- tail(df$Strat_Return, 1) - 1

cat("SPY Buy & Hold Return:", round(100 * total_return_spy, 2), "%\n")
cat("Strategy Return:", round(100 * total_return_strat, 2), "%\n")
cat("Relative Performance:", round(100 * (total_return_strat - total_return_spy), 2), "%\n")


windows()

#Plot Portfolio Performance
p1 <- ggplot(df, aes(x = Date)) +
  geom_line(aes(y = SPY_Return, color = "SPY Buy & Hold")) +
  geom_line(aes(y = Strat_Return, color = "Strategy")) +
  labs(
    title = "Golden Cross + RSI Strategy vs SPY Performance",
    y = "Cumulative Growth (normalized to 1)",
    x = "Date",
    color = "Portfolio"
  ) +
  theme_minimal()
print(p1)



#Plot Positions
p2 <- ggplot(df, aes(x = Date, y = Close)) +
  geom_line(color = "black") +
  geom_line(aes(y = SMA50), color = "blue", alpha = 0.7) +
  geom_line(aes(y = SMA200), color = "red", alpha = 0.7) +
  geom_point(data = df[df$buy_signal == TRUE, ], aes(y = Close), color = "green", size = 2, shape = 24, fill = "green") +
  geom_point(data = df[df$sell_signal == TRUE, ], aes(y = Close), color = "red", size = 2, shape = 25, fill = "red") +
  labs(title = "Buy (Green) and Sell (Red) Signals", y = "SPY Price", x = "Date") +
  theme_minimal()
#print(p2)

dbDisconnect(con)