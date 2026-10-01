// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/** @file AccountedAllocator.h
 * @brief Internal allocator with optional per-owner payload accounting.
 * Threading: the account belongs to one setup/worker operation. No global state.
 */
#include <cstddef>
#include <limits>
#include <memory>
#include <new>
#include <type_traits>

namespace dspark::detail
{

struct AllocationAccount final
{
    void *context = nullptr;
    void (*request)(void *, std::size_t) = nullptr;
    void *(*allocate)(void *, std::size_t, std::size_t) = nullptr;
    void (*deallocate)(void *, void *, std::size_t, std::size_t) = nullptr;

    void charge(std::size_t bytes) const
    {
        if (request)
            request(context, bytes);
    }
};

// Charges the actual requested payload before allocating. Accounting may reject
// the request by throwing; the underlying allocator is then never called. This
// is cumulative requested payload, not live bytes, so freeing does not refund it.
// The account must outlive its containers. A custom allocation source supplies
// both callbacks and accounts for its own upstream payload, including metadata.
template <typename T> class AccountedAllocator
{
  public:
    using value_type = T;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;
    using is_always_equal = std::false_type;

    AccountedAllocator() noexcept = default;
    explicit AccountedAllocator(const AllocationAccount *account) noexcept : account_(account) {}
    template <typename U>
    AccountedAllocator(const AccountedAllocator<U> &other) noexcept : account_(other.account()) {}

    [[nodiscard]] T *allocate(std::size_t count)
    {
        // Let the standard allocator reject an impossible element count without
        // wrapping the byte count passed to the account.
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T))
            return std::allocator<T>{}.allocate(count);
        const auto bytes = count * sizeof(T);
        if (account_)
        {
            account_->charge(bytes);
            if (account_->allocate)
                return static_cast<T *>(account_->allocate(account_->context, bytes, alignof(T)));
        }
        // Use the matching allocation primitive directly: some std::allocator
        // implementations request extra large-block alignment payload even for
        // ordinary scalar T. The request here equals the charged payload.
        if constexpr (alignof(T) > alignof(std::max_align_t))
            return static_cast<T *>(::operator new(bytes, std::align_val_t{alignof(T)}));
        else
            return static_cast<T *>(::operator new(bytes));
    }
    void deallocate(T *pointer, std::size_t count) noexcept
    {
        if (account_ && account_->deallocate)
        {
            account_->deallocate(account_->context, pointer, count * sizeof(T), alignof(T));
            return;
        }
        if constexpr (alignof(T) > alignof(std::max_align_t))
            ::operator delete(pointer, std::align_val_t{alignof(T)});
        else
            ::operator delete(pointer);
    }
    [[nodiscard]] const AllocationAccount *account() const noexcept { return account_; }
    template <typename U>
    [[nodiscard]] bool operator==(const AccountedAllocator<U> &other) const noexcept
    {
        return account_ == other.account();
    }

  private:
    const AllocationAccount *account_ = nullptr;
};

} // namespace dspark::detail
