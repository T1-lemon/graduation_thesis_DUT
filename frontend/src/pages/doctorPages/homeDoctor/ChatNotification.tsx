import MailIcon from '@mui/icons-material/Mail'
import Badge from '@mui/material/Badge'
import Popover from '@mui/material/Popover'
import Typography from '@mui/material/Typography'
import React from 'react'

type Props = {}

export default function ChatNotification({}: Props) {
  const [anchorEl, setAnchorEl] = React.useState<HTMLButtonElement | null>(null)

  const handleClick = (event: React.MouseEvent<HTMLButtonElement>) => {
    setAnchorEl(event.currentTarget)
  }

  const handleClose = () => {
    setAnchorEl(null)
  }

  const open = Boolean(anchorEl)
  const id = open ? 'simple-popover' : undefined
  return (
    <>
      <Badge
        badgeContent={4}
        color='primary'
        className='badge_notification'
        onClick={handleClick}
      >
        <MailIcon className='icon_notification' />
      </Badge>
      <Popover
        id={id}
        open={open}
        anchorEl={anchorEl}
        onClose={handleClose}
        anchorOrigin={{
          vertical: 'bottom',
          horizontal: 'left'
        }}
        transformOrigin={{
          vertical: 'top',
          horizontal: 'right'
        }}
      >
        <Typography sx={{ p: 2 }}>The content of the Popover.</Typography>
      </Popover>
    </>
  )
}
